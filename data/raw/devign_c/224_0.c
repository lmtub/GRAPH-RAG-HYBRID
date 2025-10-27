void kvm_init_irq_routing(KVMState *s)

{

    int gsi_count, i;



    gsi_count = kvm_check_extension(s, KVM_CAP_IRQ_ROUTING);

    if (gsi_count > 0) {

        unsigned int gsi_bits, i;



        /* Round up so we can search ints using ffs */

        gsi_bits = ALIGN(gsi_count, 32);

        s->used_gsi_bitmap = g_malloc0(gsi_bits / 8);

        s->gsi_count = gsi_count;



        /* Mark any over-allocated bits as already in use */

        for (i = gsi_count; i < gsi_bits; i++) {

            set_gsi(s, i);

        }

    }



    s->irq_routes = g_malloc0(sizeof(*s->irq_routes));

    s->nr_allocated_irq_routes = 0;



    if (!s->direct_msi) {

        for (i = 0; i < KVM_MSI_HASHTAB_SIZE; i++) {

            QTAILQ_INIT(&s->msi_hashtab[i]);

        }

    }



    kvm_arch_init_irq_routing(s);

}
