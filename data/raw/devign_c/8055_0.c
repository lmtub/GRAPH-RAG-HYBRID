static int timer_load(QEMUFile *f, void *opaque, int version_id)

{

    if (version_id != 1 && version_id != 2)

        return -EINVAL;

    if (cpu_ticks_enabled) {

        return -EINVAL;

    }

    cpu_ticks_offset=qemu_get_be64(f);

    ticks_per_sec=qemu_get_be64(f);

    if (version_id == 2) {

        cpu_clock_offset=qemu_get_be64(f);

    }

    return 0;

}
