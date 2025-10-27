void virtio_panic(const char *string)

{

    sclp_print(string);

    disabled_wait();

    while (1) { }

}
