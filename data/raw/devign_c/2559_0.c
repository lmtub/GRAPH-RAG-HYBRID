static int v9fs_synth_fstat(FsContext *fs_ctx, int fid_type,

                            V9fsFidOpenState *fs, struct stat *stbuf)

{

    V9fsSynthOpenState *synth_open = fs->private;

    v9fs_synth_fill_statbuf(synth_open->node, stbuf);

    return 0;

}
