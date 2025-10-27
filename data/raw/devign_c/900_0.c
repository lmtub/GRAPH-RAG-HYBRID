int qemu_v9fs_synth_mkdir(V9fsSynthNode *parent, int mode,

                          const char *name, V9fsSynthNode **result)

{

    int ret;

    V9fsSynthNode *node, *tmp;



    if (!v9fs_synth_fs) {

        return EAGAIN;

    }

    if (!name || (strlen(name) >= NAME_MAX)) {

        return EINVAL;

    }

    if (!parent) {

        parent = &v9fs_synth_root;

    }

    qemu_mutex_lock(&v9fs_synth_mutex);

    QLIST_FOREACH(tmp, &parent->child, sibling) {

        if (!strcmp(tmp->name, name)) {

            ret = EEXIST;

            goto err_out;

        }

    }

    /* Add the name */

    node = v9fs_add_dir_node(parent, mode, name, NULL, v9fs_synth_node_count++);

    v9fs_add_dir_node(node, parent->attr->mode, "..",

                      parent->attr, parent->attr->inode);

    v9fs_add_dir_node(node, node->attr->mode, ".",

                      node->attr, node->attr->inode);

    *result = node;

    ret = 0;

err_out:

    qemu_mutex_unlock(&v9fs_synth_mutex);

    return ret;

}
