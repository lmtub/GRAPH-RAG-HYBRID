int v9fs_co_open2(V9fsState *s, V9fsFidState *fidp, char *fullname, gid_t gid,

                  int flags, int mode)

{

    int err;

    FsCred cred;



    cred_init(&cred);

    cred.fc_mode = mode & 07777;

    cred.fc_uid = fidp->uid;

    cred.fc_gid = gid;

    v9fs_co_run_in_worker(

        {

            fidp->fs.fd = s->ops->open2(&s->ctx, fullname, flags, &cred);

            err = 0;

            if (fidp->fs.fd == -1) {

                err = -errno;

            }

        });

    if (!err) {

        total_open_fd++;

        if (total_open_fd > open_fd_hw) {

            v9fs_reclaim_fd(s);

        }

    }

    return err;

}
