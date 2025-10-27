AddfdInfo *qmp_add_fd(bool has_fdset_id, int64_t fdset_id, bool has_opaque,

                      const char *opaque, Error **errp)

{

    int fd;

    Monitor *mon = cur_mon;

    MonFdset *mon_fdset = NULL;

    MonFdsetFd *mon_fdset_fd;

    AddfdInfo *fdinfo;



    fd = qemu_chr_fe_get_msgfd(mon->chr);

    if (fd == -1) {

        error_set(errp, QERR_FD_NOT_SUPPLIED);

        goto error;

    }



    if (has_fdset_id) {

        QLIST_FOREACH(mon_fdset, &mon_fdsets, next) {

            /* Break if match found or match impossible due to ordering by ID */

            if (fdset_id <= mon_fdset->id) {

                if (fdset_id < mon_fdset->id) {

                    mon_fdset = NULL;

                }

                break;

            }

        }

    }



    if (mon_fdset == NULL) {

        int64_t fdset_id_prev = -1;

        MonFdset *mon_fdset_cur = QLIST_FIRST(&mon_fdsets);



        if (has_fdset_id) {

            if (fdset_id < 0) {

                error_set(errp, QERR_INVALID_PARAMETER_VALUE, "fdset-id",

                          "a non-negative value");

                goto error;

            }

            /* Use specified fdset ID */

            QLIST_FOREACH(mon_fdset, &mon_fdsets, next) {

                mon_fdset_cur = mon_fdset;

                if (fdset_id < mon_fdset_cur->id) {

                    break;

                }

            }

        } else {

            /* Use first available fdset ID */

            QLIST_FOREACH(mon_fdset, &mon_fdsets, next) {

                mon_fdset_cur = mon_fdset;

                if (fdset_id_prev == mon_fdset_cur->id - 1) {

                    fdset_id_prev = mon_fdset_cur->id;

                    continue;

                }

                break;

            }

        }



        mon_fdset = g_malloc0(sizeof(*mon_fdset));

        if (has_fdset_id) {

            mon_fdset->id = fdset_id;

        } else {

            mon_fdset->id = fdset_id_prev + 1;

        }



        /* The fdset list is ordered by fdset ID */

        if (!mon_fdset_cur) {

            QLIST_INSERT_HEAD(&mon_fdsets, mon_fdset, next);

        } else if (mon_fdset->id < mon_fdset_cur->id) {

            QLIST_INSERT_BEFORE(mon_fdset_cur, mon_fdset, next);

        } else {

            QLIST_INSERT_AFTER(mon_fdset_cur, mon_fdset, next);

        }

    }



    mon_fdset_fd = g_malloc0(sizeof(*mon_fdset_fd));

    mon_fdset_fd->fd = fd;

    mon_fdset_fd->removed = false;

    if (has_opaque) {

        mon_fdset_fd->opaque = g_strdup(opaque);

    }

    QLIST_INSERT_HEAD(&mon_fdset->fds, mon_fdset_fd, next);



    fdinfo = g_malloc0(sizeof(*fdinfo));

    fdinfo->fdset_id = mon_fdset->id;

    fdinfo->fd = mon_fdset_fd->fd;



    return fdinfo;



error:

    if (fd != -1) {

        close(fd);

    }

    return NULL;

}
