"""Central visual identity for the unified desktop application."""

from __future__ import annotations

from typing import Any


COLORS = {
    "main_background": "#020817",
    "secondary_background": "#061426",
    "card_background": "#0A2033",
    "elevated_panel": "#0D2940",
    "cyan": "#16D9E8",
    "bright_cyan": "#64E9F3",
    "dark_turquoise": "#087F91",
    "gold": "#D8B36A",
    "light_gold": "#EDCF8A",
    "main_text": "#F6F7F8",
    "secondary_text": "#AAB7C5",
    "muted_text": "#708296",
    "success": "#41D6A3",
    "warning": "#E7B85C",
    "error": "#E56A6A",
    "video_background": "#00040B",
}

FONTS = {
    "application_title": ("Segoe UI", 26, "bold"),
    "screen_heading": ("Segoe UI", 20, "bold"),
    "family_title": ("Segoe UI", 16, "bold"),
    "badge": ("Segoe UI", 10, "bold"),
    "normal": ("Segoe UI", 11),
    "small": ("Segoe UI", 10),
    "button": ("Segoe UI", 11, "bold"),
    "overlay": ("Segoe UI", 11),
    "overlay_value": ("Segoe UI", 11, "bold"),
    "dashboard_value": ("Segoe UI", 20, "bold"),
    "countdown": ("Segoe UI", 56, "bold"),
}


def configure_ttk_theme(style: Any, root: Any) -> None:
    """Configure one high-contrast Navy/Turquoise/Gold ttk theme."""

    try:
        style.theme_use("clam")
    except Exception:
        # Tk builds differ in the themes installed; style configuration below
        # remains valid with the default theme.
        pass
    style.configure("App.TFrame", background=COLORS["main_background"])
    style.configure("Secondary.TFrame", background=COLORS["secondary_background"])
    style.configure(
        "Card.TFrame",
        background=COLORS["card_background"],
        relief="solid",
        borderwidth=1,
    )
    style.configure("Panel.TFrame", background=COLORS["elevated_panel"])
    style.configure(
        "AppTitle.TLabel",
        background=COLORS["main_background"],
        foreground=COLORS["gold"],
        font=FONTS["application_title"],
    )
    style.configure(
        "Heading.TLabel",
        background=COLORS["main_background"],
        foreground=COLORS["light_gold"],
        font=FONTS["screen_heading"],
    )
    style.configure(
        "Subtitle.TLabel",
        background=COLORS["main_background"],
        foreground=COLORS["secondary_text"],
        font=FONTS["normal"],
    )
    style.configure(
        "CardTitle.TLabel",
        background=COLORS["card_background"],
        foreground=COLORS["light_gold"],
        font=FONTS["family_title"],
    )
    style.configure(
        "CardText.TLabel",
        background=COLORS["card_background"],
        foreground=COLORS["secondary_text"],
        font=FONTS["normal"],
    )
    style.configure(
        "Body.TLabel",
        background=COLORS["main_background"],
        foreground=COLORS["main_text"],
        font=FONTS["normal"],
    )
    style.configure(
        "PanelText.TLabel",
        background=COLORS["elevated_panel"],
        foreground=COLORS["main_text"],
        font=FONTS["normal"],
    )
    style.configure(
        "Metric.TFrame",
        background=COLORS["card_background"],
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "MetricName.TLabel",
        background=COLORS["card_background"],
        foreground=COLORS["secondary_text"],
        font=FONTS["small"],
    )
    style.configure(
        "MetricValue.TLabel",
        background=COLORS["card_background"],
        foreground=COLORS["bright_cyan"],
        font=FONTS["dashboard_value"],
    )
    style.configure(
        "Primary.TButton",
        background=COLORS["cyan"],
        foreground=COLORS["main_background"],
        bordercolor=COLORS["bright_cyan"],
        font=FONTS["button"],
        padding=(15, 8),
    )
    style.map(
        "Primary.TButton",
        background=[("active", COLORS["bright_cyan"]), ("disabled", COLORS["muted_text"])],
        foreground=[("disabled", COLORS["secondary_background"])],
    )
    style.configure(
        "Secondary.TButton",
        background=COLORS["elevated_panel"],
        foreground=COLORS["main_text"],
        bordercolor=COLORS["dark_turquoise"],
        font=FONTS["button"],
        padding=(13, 8),
    )
    style.map("Secondary.TButton", background=[("active", COLORS["dark_turquoise"])])
    style.configure(
        "App.Horizontal.TProgressbar",
        troughcolor=COLORS["secondary_background"],
        background=COLORS["cyan"],
        bordercolor=COLORS["secondary_background"],
    )
