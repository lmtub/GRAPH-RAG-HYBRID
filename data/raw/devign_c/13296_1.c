static void generic_loader_realize(DeviceState *dev, Error **errp)

{

    GenericLoaderState *s = GENERIC_LOADER(dev);

    hwaddr entry;

    int big_endian;

    int size = 0;



    s->set_pc = false;



    /* Perform some error checking on the user's options */

    if (s->data || s->data_len  || s->data_be) {

        /* User is loading memory values */

        if (s->file) {

            error_setg(errp, "Specifying a file is not supported when loading "

                       "memory values");

            return;

        } else if (s->force_raw) {

            error_setg(errp, "Specifying force-raw is not supported when "

                       "loading memory values");

            return;

        } else if (!s->data_len) {

            /* We can't check for !data here as a value of 0 is still valid. */

            error_setg(errp, "Both data and data-len must be specified");

            return;

        } else if (s->data_len > 8) {

            error_setg(errp, "data-len cannot be greater then 8 bytes");

            return;

        }

    } else if (s->file || s->force_raw)  {

        /* User is loading an image */

        if (s->data || s->data_len || s->data_be) {

            error_setg(errp, "data can not be specified when loading an "

                       "image");

            return;

        }

        /* The user specified a file, only set the PC if they also specified

         * a CPU to use.

         */

        if (s->cpu_num != CPU_NONE) {

            s->set_pc = true;

        }

    } else if (s->addr) {

        /* User is setting the PC */

        if (s->data || s->data_len || s->data_be) {

            error_setg(errp, "data can not be specified when setting a "

                       "program counter");

            return;

        } else if (!s->cpu_num) {

            error_setg(errp, "cpu_num must be specified when setting a "

                       "program counter");

            return;

        }

        s->set_pc = true;

    } else {

        /* Did the user specify anything? */

        error_setg(errp, "please include valid arguments");

        return;

    }



    qemu_register_reset(generic_loader_reset, dev);



    if (s->cpu_num != CPU_NONE) {

        s->cpu = qemu_get_cpu(s->cpu_num);

        if (!s->cpu) {

            error_setg(errp, "Specified boot CPU#%d is nonexistent",

                       s->cpu_num);

            return;

        }

    } else {

        s->cpu = first_cpu;

    }



#ifdef TARGET_WORDS_BIGENDIAN

    big_endian = 1;

#else

    big_endian = 0;

#endif



    if (s->file) {

        if (!s->force_raw) {

            size = load_elf_as(s->file, NULL, NULL, &entry, NULL, NULL,

                               big_endian, 0, 0, 0, s->cpu->as);



            if (size < 0) {

                size = load_uimage_as(s->file, &entry, NULL, NULL, NULL, NULL,

                                      s->cpu->as);

            }

        }



        if (size < 0 || s->force_raw) {

            /* Default to the maximum size being the machine's ram size */

            size = load_image_targphys_as(s->file, s->addr, ram_size,

                                          s->cpu->as);

        } else {

            s->addr = entry;

        }



        if (size < 0) {

            error_setg(errp, "Cannot load specified image %s", s->file);

            return;

        }

    }



    /* Convert the data endiannes */

    if (s->data_be) {

        s->data = cpu_to_be64(s->data);

    } else {

        s->data = cpu_to_le64(s->data);

    }

}
