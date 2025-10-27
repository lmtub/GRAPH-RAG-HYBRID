static int v9fs_synth_get_dentry(V9fsSynthNode *dir, struct dirent *entry,

                                 struct dirent **result, off_t off)

{

    int i = 0;

    V9fsSynthNode *node;



    rcu_read_lock();

    QLIST_FOREACH(node, &dir->child, sibling) {

        /* This is the off child of the directory */

        if (i == off) {

            break;

        }

        i++;

    }

    rcu_read_unlock();

    if (!node) {

        /* end of directory */

        *result = NULL;

        return 0;

    }

    v9fs_synth_direntry(node, entry, off);

    *result = entry;

    return 0;

}
