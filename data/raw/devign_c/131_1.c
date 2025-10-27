static void device_finalize(Object *obj)

{

    NamedGPIOList *ngl, *next;



    DeviceState *dev = DEVICE(obj);

    qemu_opts_del(dev->opts);



    QLIST_FOREACH_SAFE(ngl, &dev->gpios, node, next) {

        QLIST_REMOVE(ngl, node);

        qemu_free_irqs(ngl->in, ngl->num_in);

        g_free(ngl->name);

        g_free(ngl);

        /* ngl->out irqs are owned by the other end and should not be freed

         * here

         */

    }

}
