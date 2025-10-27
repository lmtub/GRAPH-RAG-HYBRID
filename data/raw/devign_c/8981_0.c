static int v9fs_synth_readdir_r(FsContext *ctx, V9fsFidOpenState *fs,

                                struct dirent *entry, struct dirent **result)

{

    int ret;

    V9fsSynthOpenState *synth_open = fs->private;

    V9fsSynthNode *node = synth_open->node;

    ret = v9fs_synth_get_dentry(node, entry, result, synth_open->offset);

    if (!ret && *result != NULL) {

        synth_open->offset++;

    }

    return ret;

}
