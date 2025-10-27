static void tm_put(QEMUFile *f, struct tm *tm) {

    qemu_put_be16(f, tm->tm_sec);

    qemu_put_be16(f, tm->tm_min);

    qemu_put_be16(f, tm->tm_hour);

    qemu_put_be16(f, tm->tm_mday);

    qemu_put_be16(f, tm->tm_min);

    qemu_put_be16(f, tm->tm_year);

}
