# Persistent native voice server — Tier 2 of the hybrid audio engine.
#
# Started once and kept alive, so no utterance pays the ~1.9 s PowerShell +
# System.Speech cold start. Protocol, one line per request on stdin:
#
#     <lang>|<text>     speak <text> using the best installed voice for <lang>
#     ?INVENTORY        report the discovered voices, then continue
#     __QUIT__          shut down
#
# Replies: DONE (finished speaking) | NOVOICE (no voice for that language)
#          FAIL (exception) | INV|... (inventory listing)
#
# The DONE handshake is what lets Python hold its completion lock for the exact
# duration of the audio rather than guessing at phrase length.
#
# VOICE RESOLUTION IS BY LANGUAGE, NOT BY NAME.
# An earlier revision hardcoded one voice name per language ("Naayf" for Arabic).
# That is brittle in both directions: it fails when that specific voice is absent
# even though another Arabic voice is installed, and it would happily match a
# same-named voice of the wrong language. Resolution now asks the real question —
# "which installed voice can actually speak this language?" — by testing each
# voice's culture, and only falls back to name hints as a tie-breaker for ordering.
#
# Speech is synchronous on purpose. An earlier revision called SpeakAsyncCancelAll
# before each phrase (barge-in); combined with a prompt repeating faster than it
# could be spoken, that restarted the phrase forever and never got past its first
# two words. Phrases now run to completion and the caller drops cues mid-speech.

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 1

# --- Discover what this machine can actually speak ---------------------------
# name -> culture, for every voice System.Speech can drive.
$voices = @()
try {
    foreach ($v in $synth.GetInstalledVoices()) {
        if (-not $v.Enabled) { continue }
        $voices += [PSCustomObject]@{
            Name    = $v.VoiceInfo.Name
            Culture = $v.VoiceInfo.Culture.Name      # e.g. "en-US", "ar-SA"
            Lang    = $v.VoiceInfo.Culture.TwoLetterISOLanguageName
        }
    }
} catch { }

# OneCore voices live in a separate registry hive and are NOT reachable through
# System.Speech. They are enumerated here for diagnostics only, so an operator can
# see that (say) an Arabic OneCore voice exists but needs its desktop counterpart
# installed. Bridging the hives requires writing to HKLM, which this script will
# not do on its own.
$oneCore = @()
try {
    $p = "HKLM:\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens"
    if (Test-Path $p) {
        foreach ($k in Get-ChildItem $p -ErrorAction SilentlyContinue) {
            $nm = (Get-ItemProperty $k.PSPath -ErrorAction SilentlyContinue).'(default)'
            if ($nm) { $oneCore += $nm }
        }
    }
} catch { }

# Preference order when a language has several usable voices. Purely a tie-break;
# a voice absent from this list is still eligible if its culture matches.
$preferred = @{
    "ar" = @("Naayf", "Hoda", "Salma", "Hamed", "Zariyah")
    "en" = @("Zira", "Aria", "David", "Mark")
}

$currentLang = ""
$currentOk = $false

function Resolve-VoiceForLang([string]$lang) {
    if ([string]::IsNullOrWhiteSpace($lang)) { return $true }
    if ($lang -eq $script:currentLang) { return $script:currentOk }

    $lang = $lang.Trim().ToLower()
    $candidates = @($script:voices | Where-Object { $_.Lang -eq $lang })

    if ($candidates.Count -eq 0) {
        $script:currentLang = $lang
        $script:currentOk = $false
        return $false
    }

    # Order by preference list, then by whatever remains.
    $ordered = @()
    if ($script:preferred.ContainsKey($lang)) {
        foreach ($hint in $script:preferred[$lang]) {
            $ordered += @($candidates | Where-Object { $_.Name -like "*$hint*" })
        }
    }
    $ordered += @($candidates | Where-Object { $ordered -notcontains $_ })

    foreach ($v in $ordered) {
        try {
            $script:synth.SelectVoice($v.Name)
            $script:currentLang = $lang
            $script:currentOk = $true
            return $true
        } catch { }
    }

    $script:currentLang = $lang
    $script:currentOk = $false
    return $false
}

while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) { break }              # stdin closed -> parent exited
    $line = $line.Trim()
    if ($line -eq '') { continue }
    if ($line -eq '__QUIT__') { break }

    # ?RESOLVE|<lang> -> the voice that WOULD be used, or NOVOICE. Reporting the
    # resolver's own answer avoids a caller guessing from the inventory and naming
    # a different voice than the one that actually speaks.
    if ($line.StartsWith('?RESOLVE|')) {
        $q = $line.Substring(9).Trim()
        if (Resolve-VoiceForLang $q) {
            [Console]::Out.WriteLine("VOICE|" + $synth.Voice.Name)
        } else {
            [Console]::Out.WriteLine("NOVOICE")
        }
        [Console]::Out.Flush()
        continue
    }

    if ($line -eq '?INVENTORY') {
        $sapi = ($voices | ForEach-Object { "$($_.Name)[$($_.Culture)]" }) -join ";"
        $oc = ($oneCore -join ";")
        [Console]::Out.WriteLine("INV|sapi=$sapi|onecore=$oc")
        [Console]::Out.Flush()
        continue
    }

    $lang = ""
    $text = $line
    $sep = $line.IndexOf('|')
    if ($sep -ge 0) {
        $lang = $line.Substring(0, $sep).Trim()
        $text = $line.Substring($sep + 1).Trim()
    }
    if ($text -eq '') { continue }

    try {
        if (-not (Resolve-VoiceForLang $lang)) {
            # No installed voice speaks this language. Stay silent rather than
            # reading the text with a wrong-language voice, which yields gibberish;
            # Python then falls through to its chime tier.
            [Console]::Out.WriteLine("NOVOICE")
        } else {
            $synth.Speak($text)                 # synchronous: runs to completion
            [Console]::Out.WriteLine("DONE")
        }
    } catch {
        [Console]::Out.WriteLine("FAIL")
    }
    [Console]::Out.Flush()
}

try { $synth.Dispose() } catch { }
