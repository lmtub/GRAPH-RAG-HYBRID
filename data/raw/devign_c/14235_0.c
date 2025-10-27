static void hmp_cont_cb(void *opaque, int err)

{

    Monitor *mon = opaque;



    if (!err) {

        hmp_cont(mon, NULL);

    }

}
