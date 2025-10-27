void gtod_save(QEMUFile *f, void *opaque)

{

    uint64_t tod_low;

    uint8_t tod_high;

    int r;



    r = s390_get_clock(&tod_high, &tod_low);

    if (r) {

        fprintf(stderr, "WARNING: Unable to get guest clock for migration. "

                        "Error code %d. Guest clock will not be migrated "

                        "which could cause the guest to hang.\n", r);

        qemu_put_byte(f, S390_TOD_CLOCK_VALUE_MISSING);

        return;

    }



    qemu_put_byte(f, S390_TOD_CLOCK_VALUE_PRESENT);

    qemu_put_byte(f, tod_high);

    qemu_put_be64(f, tod_low);

}
