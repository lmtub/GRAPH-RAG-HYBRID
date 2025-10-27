static void input_linux_complete(UserCreatable *uc, Error **errp)

{

    InputLinux *il = INPUT_LINUX(uc);

    uint8_t evtmap, relmap, absmap, keymap[KEY_CNT / 8];

    unsigned int i;

    int rc, ver;



    if (!il->evdev) {

        error_setg(errp, "no input device specified");

        return;

    }



    il->fd = open(il->evdev, O_RDWR);

    if (il->fd < 0)  {

        error_setg_file_open(errp, errno, il->evdev);

        return;

    }

    qemu_set_nonblock(il->fd);



    rc = ioctl(il->fd, EVIOCGVERSION, &ver);

    if (rc < 0) {

        error_setg(errp, "%s: is not an evdev device", il->evdev);

        goto err_close;

    }



    rc = ioctl(il->fd, EVIOCGBIT(0, sizeof(evtmap)), &evtmap);

    if (rc < 0) {

        error_setg(errp, "%s: failed to read event bits", il->evdev);

        goto err_close;

    }



    if (evtmap & (1 << EV_REL)) {

        relmap = 0;

        rc = ioctl(il->fd, EVIOCGBIT(EV_REL, sizeof(relmap)), &relmap);

        if (relmap & (1 << REL_X)) {

            il->has_rel_x = true;

        }

    }



    if (evtmap & (1 << EV_ABS)) {

        absmap = 0;

        rc = ioctl(il->fd, EVIOCGBIT(EV_ABS, sizeof(absmap)), &absmap);

        if (absmap & (1 << ABS_X)) {

            il->has_abs_x = true;

        }

    }



    if (evtmap & (1 << EV_KEY)) {

        memset(keymap, 0, sizeof(keymap));

        rc = ioctl(il->fd, EVIOCGBIT(EV_KEY, sizeof(keymap)), keymap);

        for (i = 0; i < KEY_CNT; i++) {

            if (keymap[i / 8] & (1 << (i % 8))) {

                if (linux_is_button(i)) {

                    il->num_btns++;

                } else {

                    il->num_keys++;

                }

            }

        }

    }



    qemu_set_fd_handler(il->fd, input_linux_event, NULL, il);

    input_linux_toggle_grab(il);

    QTAILQ_INSERT_TAIL(&inputs, il, next);

    il->initialized = true;

    return;



err_close:

    close(il->fd);

    return;

}
