static void kvm_add_routing_entry(KVMState *s,

                                  struct kvm_irq_routing_entry *entry)

{

    struct kvm_irq_routing_entry *new;

    int n, size;



    if (s->irq_routes->nr == s->nr_allocated_irq_routes) {

        n = s->nr_allocated_irq_routes * 2;

        if (n < 64) {

            n = 64;

        }

        size = sizeof(struct kvm_irq_routing);

        size += n * sizeof(*new);

        s->irq_routes = g_realloc(s->irq_routes, size);

        s->nr_allocated_irq_routes = n;

    }

    n = s->irq_routes->nr++;

    new = &s->irq_routes->entries[n];

    memset(new, 0, sizeof(*new));

    new->gsi = entry->gsi;

    new->type = entry->type;

    new->flags = entry->flags;

    new->u = entry->u;



    set_gsi(s, entry->gsi);

}
