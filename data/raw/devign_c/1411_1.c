static void coroutine_fn v9fs_xattrwalk(void *opaque)

{

    int64_t size;

    V9fsString name;

    ssize_t err = 0;

    size_t offset = 7;

    int32_t fid, newfid;

    V9fsFidState *file_fidp;

    V9fsFidState *xattr_fidp = NULL;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    v9fs_string_init(&name);

    err = pdu_unmarshal(pdu, offset, "dds", &fid, &newfid, &name);

    if (err < 0) {

        goto out_nofid;

    }

    trace_v9fs_xattrwalk(pdu->tag, pdu->id, fid, newfid, name.data);



    file_fidp = get_fid(pdu, fid);

    if (file_fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }

    xattr_fidp = alloc_fid(s, newfid);

    if (xattr_fidp == NULL) {

        err = -EINVAL;

        goto out;

    }

    v9fs_path_copy(&xattr_fidp->path, &file_fidp->path);

    if (!v9fs_string_size(&name)) {

        /*

         * listxattr request. Get the size first

         */

        size = v9fs_co_llistxattr(pdu, &xattr_fidp->path, NULL, 0);

        if (size < 0) {

            err = size;

            clunk_fid(s, xattr_fidp->fid);

            goto out;

        }

        /*

         * Read the xattr value

         */

        xattr_fidp->fs.xattr.len = size;

        xattr_fidp->fid_type = P9_FID_XATTR;

        xattr_fidp->fs.xattr.xattrwalk_fid = true;

        if (size) {

            xattr_fidp->fs.xattr.value = g_malloc(size);

            err = v9fs_co_llistxattr(pdu, &xattr_fidp->path,

                                     xattr_fidp->fs.xattr.value,

                                     xattr_fidp->fs.xattr.len);

            if (err < 0) {

                clunk_fid(s, xattr_fidp->fid);

                goto out;

            }

        }

        err = pdu_marshal(pdu, offset, "q", size);

        if (err < 0) {

            goto out;

        }

        err += offset;

    } else {

        /*

         * specific xattr fid. We check for xattr

         * presence also collect the xattr size

         */

        size = v9fs_co_lgetxattr(pdu, &xattr_fidp->path,

                                 &name, NULL, 0);

        if (size < 0) {

            err = size;

            clunk_fid(s, xattr_fidp->fid);

            goto out;

        }

        /*

         * Read the xattr value

         */

        xattr_fidp->fs.xattr.len = size;

        xattr_fidp->fid_type = P9_FID_XATTR;

        xattr_fidp->fs.xattr.xattrwalk_fid = true;

        if (size) {

            xattr_fidp->fs.xattr.value = g_malloc(size);

            err = v9fs_co_lgetxattr(pdu, &xattr_fidp->path,

                                    &name, xattr_fidp->fs.xattr.value,

                                    xattr_fidp->fs.xattr.len);

            if (err < 0) {

                clunk_fid(s, xattr_fidp->fid);

                goto out;

            }

        }

        err = pdu_marshal(pdu, offset, "q", size);

        if (err < 0) {

            goto out;

        }

        err += offset;

    }

    trace_v9fs_xattrwalk_return(pdu->tag, pdu->id, size);

out:

    put_fid(pdu, file_fidp);

    if (xattr_fidp) {

        put_fid(pdu, xattr_fidp);

    }

out_nofid:

    pdu_complete(pdu, err);

    v9fs_string_free(&name);

}
