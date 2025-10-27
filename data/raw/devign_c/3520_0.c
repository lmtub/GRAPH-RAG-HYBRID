static int v9fs_synth_name_to_path(FsContext *ctx, V9fsPath *dir_path,

                                   const char *name, V9fsPath *target)

{

    V9fsSynthNode *node;

    V9fsSynthNode *dir_node;



    /* "." and ".." are not allowed */

    if (!strcmp(name, ".") || !strcmp(name, "..")) {

        errno = EINVAL;

        return -1;



    }

    if (!dir_path) {

        dir_node = &v9fs_synth_root;

    } else {

        dir_node = *(V9fsSynthNode **)dir_path->data;

    }

    if (!strcmp(name, "/")) {

        node = dir_node;

        goto out;

    }

    /* search for the name in the childern */

    rcu_read_lock();

    QLIST_FOREACH(node, &dir_node->child, sibling) {

        if (!strcmp(node->name, name)) {

            break;

        }

    }

    rcu_read_unlock();



    if (!node) {

        errno = ENOENT;

        return -1;

    }

out:

    /* Copy the node pointer to fid */

    target->data = g_malloc(sizeof(void *));

    memcpy(target->data, &node, sizeof(void *));

    target->size = sizeof(void *);

    return 0;

}
