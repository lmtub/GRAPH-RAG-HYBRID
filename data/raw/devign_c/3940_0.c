static KVMSlot *kvm_alloc_slot(KVMState *s)

{

    int i;



    for (i = 0; i < ARRAY_SIZE(s->slots); i++) {

        /* KVM private memory slots */

        if (i >= 8 && i < 12)

            continue;

        if (s->slots[i].memory_size == 0)

            return &s->slots[i];

    }



    fprintf(stderr, "%s: no free slot available\n", __func__);

    abort();

}
