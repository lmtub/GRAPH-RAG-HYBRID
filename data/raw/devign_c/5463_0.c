static int qemu_savevm_state(Monitor *mon, QEMUFile *f)

{

    int ret;



    if (qemu_savevm_state_blocked(mon)) {

        ret = -EINVAL;

        goto out;

    }



    ret = qemu_savevm_state_begin(f, 0, 0);

    if (ret < 0)

        goto out;



    do {

        ret = qemu_savevm_state_iterate(f);

        if (ret < 0)

            goto out;

    } while (ret == 0);



    ret = qemu_savevm_state_complete(f);



out:

    if (ret == 0) {

        ret = qemu_file_get_error(f);

    }



    return ret;

}
