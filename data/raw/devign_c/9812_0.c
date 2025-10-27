static off_t v9fs_synth_telldir(FsContext *ctx, V9fsFidOpenState *fs)

{

    V9fsSynthOpenState *synth_open = fs->private;

    return synth_open->offset;

}
