# reMarkable 2 setup

## Build and copy

Requires Docker and SSH access to the tablet. Replace `remar` with your SSH host.
Download a model first using the [usage instructions](usage.md).

```sh
make tablet-image  # build the cross-compilation image once
make tablet        # produces build/unremarkable-armv7
ssh remar 'mkdir -p /home/root/unremarkable'
scp build/unremarkable-armv7 remar:/home/root/unremarkable/unremarkable
scp -r models/ device/ remar:/home/root/unremarkable/
```

Store the engine and models under `/home/root`; the tablet's root filesystem
has very little free space. Copy only the model files you need.

The cross-build targets ARMv7 hard-float with NEON/VFPv4, using Ubuntu 24.04.
It links libstdc++ and libgcc statically to avoid the tablet's older C++ runtime.
`-ffp-contract=off` prevents floating-point contraction from changing results
against the reference fixture. See the [Makefile](../Makefile) for exact flags.

## Optional UI patches

| File | Purpose |
| --- | --- |
| [ask.sh](../device/ask.sh) | Runs a preset or free-text subject, streaming into `state/` |
| [ask-bg.sh](../device/ask-bg.sh) | Runs inference in the background |
| [unremarkable.qmd](../device/unremarkable.qmd) | Quick Settings panel with presets |
| [unremarkable-selection.qmd](../device/unremarkable-selection.qmd) | Generates a poem from a handwriting selection |

The `.qmd` files use [qmldiff](https://github.com/asivery/qmldiff) through
qt-resource-rebuilder under [xovi](https://github.com/asivery/xovi).
Install dependencies on the tablet with:

```sh
vellum add xovi qt-resource-rebuilder qt-command-executor
```

Copy the patches to `/home/root/xovi/exthome/qt-resource-rebuilder/`.
Run `/home/root/xovi/start` to apply them or `/home/root/xovi/stock` to revert.
The systemd override lives on tmpfs: rebooting restores the stock UI, and you
must start the patches again.

The selection patch uses the device's handwriting recognizer and passes the
recognized text to `ask.sh` as an argument. Avoid regular-expression literals
inside `.qmd` files: qmldiff can misread embedded quotes and corrupt the QML.
