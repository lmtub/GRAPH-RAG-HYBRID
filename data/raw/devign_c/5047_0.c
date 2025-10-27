static int local_set_mapped_file_attrat(int dirfd, const char *name,

                                        FsCred *credp)

{

    FILE *fp;

    int ret;

    char buf[ATTR_MAX];

    int uid = -1, gid = -1, mode = -1, rdev = -1;

    int map_dirfd;



    ret = mkdirat(dirfd, VIRTFS_META_DIR, 0700);

    if (ret < 0 && errno != EEXIST) {

        return -1;

    }



    map_dirfd = openat_dir(dirfd, VIRTFS_META_DIR);

    if (map_dirfd == -1) {

        return -1;

    }



    fp = local_fopenat(map_dirfd, name, "r");

    if (!fp) {

        if (errno == ENOENT) {

            goto update_map_file;

        } else {

            close_preserve_errno(map_dirfd);

            return -1;

        }

    }

    memset(buf, 0, ATTR_MAX);

    while (fgets(buf, ATTR_MAX, fp)) {

        if (!strncmp(buf, "virtfs.uid", 10)) {

            uid = atoi(buf + 11);

        } else if (!strncmp(buf, "virtfs.gid", 10)) {

            gid = atoi(buf + 11);

        } else if (!strncmp(buf, "virtfs.mode", 11)) {

            mode = atoi(buf + 12);

        } else if (!strncmp(buf, "virtfs.rdev", 11)) {

            rdev = atoi(buf + 12);

        }

        memset(buf, 0, ATTR_MAX);

    }

    fclose(fp);



update_map_file:

    fp = local_fopenat(map_dirfd, name, "w");

    close_preserve_errno(map_dirfd);

    if (!fp) {

        return -1;

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



    return 0;

}
