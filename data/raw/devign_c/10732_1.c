size_t qemu_file_set_rate_limit(QEMUFile *f, size_t new_rate)

{

    if (f->set_rate_limit)

        return f->set_rate_limit(f->opaque, new_rate);



    return 0;

}
