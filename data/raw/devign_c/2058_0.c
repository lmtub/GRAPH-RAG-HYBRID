static void v9fs_synth_seekdir(FsContext *ctx, V9fsFidOpenState *fs, off_t off)

{

    V9fsSynthOpenState *synth_open = fs->private;

    synth_open->offset = off;

}
