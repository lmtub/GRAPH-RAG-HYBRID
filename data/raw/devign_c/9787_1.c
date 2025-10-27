GIOStatus ga_channel_write_all(GAChannel *c, const char *buf, size_t size)

{

    GIOStatus status = G_IO_STATUS_NORMAL;

    size_t count;



    while (size) {

        status = ga_channel_write(c, buf, size, &count);

        if (status == G_IO_STATUS_NORMAL) {

            size -= count;

            buf += count;

        } else if (status != G_IO_STATUS_AGAIN) {

            break;

        }

    }



    return status;

}
