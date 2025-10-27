static int raw_eject(BlockDriverState *bs, int eject_flag)

{

    BDRVRawState *s = bs->opaque;



    switch(s->type) {

    case FTYPE_CD:

        if (eject_flag) {

            if (ioctl (s->fd, CDROMEJECT, NULL) < 0)

                perror("CDROMEJECT");

        } else {

            if (ioctl (s->fd, CDROMCLOSETRAY, NULL) < 0)

                perror("CDROMEJECT");

        }

        break;

    case FTYPE_FD:

        {

            int fd;

            if (s->fd >= 0) {

                close(s->fd);

                s->fd = -1;

                raw_close_fd_pool(s);

            }

            fd = open(bs->filename, s->fd_open_flags | O_NONBLOCK);

            if (fd >= 0) {

                if (ioctl(fd, FDEJECT, 0) < 0)

                    perror("FDEJECT");

                close(fd);

            }

        }

        break;

    default:

        return -ENOTSUP;

    }

    return 0;

}
