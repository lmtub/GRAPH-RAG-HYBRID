int64_t qmp_guest_fsfreeze_thaw(Error **err)

{

    int ret;

    GuestFsfreezeMountList mounts;

    GuestFsfreezeMount *mount;

    int fd, i = 0, logged;



    QTAILQ_INIT(&mounts);

    ret = guest_fsfreeze_build_mount_list(&mounts);

    if (ret) {

        error_set(err, QERR_QGA_COMMAND_FAILED,

                  "failed to enumerate filesystems");

        return 0;

    }



    QTAILQ_FOREACH(mount, &mounts, next) {

        logged = false;

        fd = qemu_open(mount->dirname, O_RDONLY);

        if (fd == -1) {

            continue;

        }

        /* we have no way of knowing whether a filesystem was actually unfrozen

         * as a result of a successful call to FITHAW, only that if an error

         * was returned the filesystem was *not* unfrozen by that particular

         * call.

         *

         * since multiple preceeding FIFREEZEs require multiple calls to FITHAW

         * to unfreeze, continuing issuing FITHAW until an error is returned,

         * in which case either the filesystem is in an unfreezable state, or,

         * more likely, it was thawed previously (and remains so afterward).

         *

         * also, since the most recent successful call is the one that did

         * the actual unfreeze, we can use this to provide an accurate count

         * of the number of filesystems unfrozen by guest-fsfreeze-thaw, which

         * may * be useful for determining whether a filesystem was unfrozen

         * during the freeze/thaw phase by a process other than qemu-ga.

         */

        do {

            ret = ioctl(fd, FITHAW);

            if (ret == 0 && !logged) {

                i++;

                logged = true;

            }

        } while (ret == 0);

        close(fd);

    }



    guest_fsfreeze_state.status = GUEST_FSFREEZE_STATUS_THAWED;

    enable_logging();

    guest_fsfreeze_free_mount_list(&mounts);

    return i;

}
