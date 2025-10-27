static V9fsSynthNode *v9fs_add_dir_node(V9fsSynthNode *parent, int mode,

                                        const char *name,

                                        V9fsSynthNodeAttr *attr, int inode)

{

    V9fsSynthNode *node;



    /* Add directory type and remove write bits */

    mode = ((mode & 0777) | S_IFDIR) & ~(S_IWUSR | S_IWGRP | S_IWOTH);

    node = g_malloc0(sizeof(V9fsSynthNode));

    if (attr) {

        /* We are adding .. or . entries */

        node->attr = attr;

        node->attr->nlink++;

    } else {

        node->attr = &node->actual_attr;

        node->attr->inode = inode;

        node->attr->nlink = 1;

        /* We don't allow write to directories */

        node->attr->mode   = mode;

        node->attr->write = NULL;

        node->attr->read  = NULL;

    }

    node->private = node;

    pstrcpy(node->name, sizeof(node->name), name);

    QLIST_INSERT_HEAD_RCU(&parent->child, node, sibling);

    return node;

}
