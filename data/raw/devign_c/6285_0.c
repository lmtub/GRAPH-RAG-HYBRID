int v9fs_co_mknod(V9fsState *s, V9fsString *path, uid_t uid,

                  gid_t gid, dev_t dev, mode_t mode)

{

    int err;

    FsCred cred;



    cred_init(&cred);

    cred.fc_uid  = uid;

    cred.fc_gid  = gid;

    cred.fc_mode = mode;

    cred.fc_rdev = dev;

    v9fs_co_run_in_worker(

        {

            err = s->ops->mknod(&s->ctx, path->data, &cred);

            if (err < 0) {

                err = -errno;

            }

        });

    return err;

}
