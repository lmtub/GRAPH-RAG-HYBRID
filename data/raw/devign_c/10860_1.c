static int coroutine_fn v9fs_complete_renameat(V9fsPDU *pdu, int32_t olddirfid,

                                               V9fsString *old_name,

                                               int32_t newdirfid,

                                               V9fsString *new_name)

{

    int err = 0;

    V9fsState *s = pdu->s;

    V9fsFidState *newdirfidp = NULL, *olddirfidp = NULL;



    olddirfidp = get_fid(pdu, olddirfid);

    if (olddirfidp == NULL) {

        err = -ENOENT;

        goto out;

    }

    if (newdirfid != -1) {

        newdirfidp = get_fid(pdu, newdirfid);

        if (newdirfidp == NULL) {

            err = -ENOENT;

            goto out;

        }

    } else {

        newdirfidp = get_fid(pdu, olddirfid);

    }



    err = v9fs_co_renameat(pdu, &olddirfidp->path, old_name,

                           &newdirfidp->path, new_name);

    if (err < 0) {

        goto out;

    }

    if (s->ctx.export_flags & V9FS_PATHNAME_FSCONTEXT) {

        /* Only for path based fid  we need to do the below fixup */

        v9fs_fix_fid_paths(pdu, &olddirfidp->path, old_name,

                           &newdirfidp->path, new_name);

    }

out:

    if (olddirfidp) {

        put_fid(pdu, olddirfidp);

    }

    if (newdirfidp) {

        put_fid(pdu, newdirfidp);

    }

    return err;

}
