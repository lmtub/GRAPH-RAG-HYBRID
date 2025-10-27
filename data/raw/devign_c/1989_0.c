int v9fs_co_mkdir(V9fsState *s, char *name, mode_t mode, uid_t uid, gid_t gid)

{

    int err;

    FsCred cred;



    cred_init(&cred);

    cred.fc_mode = mode;

    cred.fc_uid = uid;

    cred.fc_gid = gid;

    v9fs_co_run_in_worker(

        {

            err = s->ops->mkdir(&s->ctx, name, &cred);

            if (err < 0) {

                err = -errno;

            }

        });

    return err;

}
