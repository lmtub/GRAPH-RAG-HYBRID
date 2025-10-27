static int coroutine_fn v9fs_complete_rename(V9fsPDU *pdu, V9fsFidState *fidp,

                                             int32_t newdirfid,

                                             V9fsString *name)

{

    char *end;

    int err = 0;

    V9fsPath new_path;

    V9fsFidState *tfidp;

    V9fsState *s = pdu->s;

    V9fsFidState *dirfidp = NULL;

    char *old_name, *new_name;



    v9fs_path_init(&new_path);

    if (newdirfid != -1) {

        dirfidp = get_fid(pdu, newdirfid);

        if (dirfidp == NULL) {

            err = -ENOENT;

            goto out_nofid;

        }

        BUG_ON(dirfidp->fid_type != P9_FID_NONE);

        v9fs_co_name_to_path(pdu, &dirfidp->path, name->data, &new_path);

    } else {

        old_name = fidp->path.data;

        end = strrchr(old_name, '/');

        if (end) {

            end++;

        } else {

            end = old_name;

        }

        new_name = g_malloc0(end - old_name + name->size + 1);

        strncat(new_name, old_name, end - old_name);

        strncat(new_name + (end - old_name), name->data, name->size);

        v9fs_co_name_to_path(pdu, NULL, new_name, &new_path);

        g_free(new_name);

    }

    err = v9fs_co_rename(pdu, &fidp->path, &new_path);

    if (err < 0) {

        goto out;

    }

    /*

     * Fixup fid's pointing to the old name to

     * start pointing to the new name

     */

    for (tfidp = s->fid_list; tfidp; tfidp = tfidp->next) {

        if (v9fs_path_is_ancestor(&fidp->path, &tfidp->path)) {

            /* replace the name */

            v9fs_fix_path(&tfidp->path, &new_path, strlen(fidp->path.data));

        }

    }

out:

    if (dirfidp) {

        put_fid(pdu, dirfidp);

    }

    v9fs_path_free(&new_path);

out_nofid:

    return err;

}
