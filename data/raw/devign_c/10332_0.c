static int local_set_mapped_file_attr(FsContext *ctx,

                                      const char *path, FsCred *credp)

{

    FILE *fp;

    int ret = 0;

    char buf[ATTR_MAX];

    char attr_path[PATH_MAX];

    int uid = -1, gid = -1, mode = -1, rdev = -1;



    fp = local_fopen(local_mapped_attr_path(ctx, path, attr_path), "r");

    if (!fp) {

        goto create_map_file;

    }

    memset(buf, 0, ATTR_MAX);

    while (fgets(buf, ATTR_MAX, fp)) {

        if (!strncmp(buf, "virtfs.uid", 10)) {

            uid = atoi(buf+11);

        } else if (!strncmp(buf, "virtfs.gid", 10)) {

            gid = atoi(buf+11);

        } else if (!strncmp(buf, "virtfs.mode", 11)) {

            mode = atoi(buf+12);

        } else if (!strncmp(buf, "virtfs.rdev", 11)) {

            rdev = atoi(buf+12);

        }

        memset(buf, 0, ATTR_MAX);

    }

    fclose(fp);

    goto update_map_file;



create_map_file:

    ret = local_create_mapped_attr_dir(ctx, path);

    if (ret < 0) {

        goto err_out;

    }



update_map_file:

    fp = local_fopen(attr_path, "w");

    if (!fp) {

        ret = -1;

        goto err_out;

    }



    if (credp->fc_uid != -1) {

        uid = credp->fc_uid;

    }

    if (credp->fc_gid != -1) {

        gid = credp->fc_gid;

    }

    if (credp->fc_mode != -1) {

        mode = credp->fc_mode;

    }

    if (credp->fc_rdev != -1) {

        rdev = credp->fc_rdev;

    }





    if (uid != -1) {

        fprintf(fp, "virtfs.uid=%d\n", uid);

    }

    if (gid != -1) {

        fprintf(fp, "virtfs.gid=%d\n", gid);

    }

    if (mode != -1) {

        fprintf(fp, "virtfs.mode=%d\n", mode);

    }

    if (rdev != -1) {

        fprintf(fp, "virtfs.rdev=%d\n", rdev);

    }

    fclose(fp);



err_out:

    return ret;

}
