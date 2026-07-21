# Umber release assets

Branding assets consumed by the installer pipelines (Windows MSI via cargo-wix,
Linux AppImage via linuxdeploy). Regenerate with the tools on a dev box:

```sh
# PNG marks (AppImage / . desktop Icon source)
rsvg-convert -w 1024 -h 1024 icon.svg -o umber-1024.png
rsvg-convert -w 512  -h 512  icon.svg -o umber-512.png
rsvg-convert -w 256  -h 256  icon.svg -o umber-256.png
# Multi-size Windows icon (consumed by the MSI wxs)
magick umber-256.png -background none \
  -define icon:auto-resize=256,128,64,48,32,16 umber.ico
```

| File            | Used by                                   |
| --------------- | ----------------------------------------- |
| `icon.svg`      | Source of truth for the mark              |
| `umber-512.png` | AppImage icon → `Icon=umber` / `.DirIcon` |
| `umber.ico`      | Windows MSI `@@ICON_FILE@@`               |
| `umber.desktop` | AppImage desktop entry                    |

This placeholder mark was generated for `v0.1.0-alpha.1`; replace `icon.svg`
and rerun the commands above to rebrand.