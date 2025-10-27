static void v9fs_synth_rewinddir(FsContext *ctx, V9fsFidOpenState *fs)

{

    v9fs_synth_seekdir(ctx, fs, 0);

}
