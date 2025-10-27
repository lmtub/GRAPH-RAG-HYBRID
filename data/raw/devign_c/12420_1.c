host_memory_backend_memory_complete(UserCreatable *uc, Error **errp)

{

    HostMemoryBackend *backend = MEMORY_BACKEND(uc);

    HostMemoryBackendClass *bc = MEMORY_BACKEND_GET_CLASS(uc);

    Error *local_err = NULL;

    void *ptr;

    uint64_t sz;



    if (bc->alloc) {

        bc->alloc(backend, &local_err);

        if (local_err) {

            goto out;

        }



        ptr = memory_region_get_ram_ptr(&backend->mr);

        sz = memory_region_size(&backend->mr);



        if (backend->merge) {

            qemu_madvise(ptr, sz, QEMU_MADV_MERGEABLE);

        }

        if (!backend->dump) {

            qemu_madvise(ptr, sz, QEMU_MADV_DONTDUMP);

        }

#ifdef CONFIG_NUMA

        unsigned long lastbit = find_last_bit(backend->host_nodes, MAX_NODES);

        /* lastbit == MAX_NODES means maxnode = 0 */

        unsigned long maxnode = (lastbit + 1) % (MAX_NODES + 1);

        /* ensure policy won't be ignored in case memory is preallocated

         * before mbind(). note: MPOL_MF_STRICT is ignored on hugepages so

         * this doesn't catch hugepage case. */

        unsigned flags = MPOL_MF_STRICT | MPOL_MF_MOVE;



        /* check for invalid host-nodes and policies and give more verbose

         * error messages than mbind(). */

        if (maxnode && backend->policy == MPOL_DEFAULT) {

            error_setg(errp, "host-nodes must be empty for policy default,"

                       " or you should explicitly specify a policy other"

                       " than default");

            return;

        } else if (maxnode == 0 && backend->policy != MPOL_DEFAULT) {

            error_setg(errp, "host-nodes must be set for policy %s",

                       HostMemPolicy_lookup[backend->policy]);

            return;

        }



        /* We can have up to MAX_NODES nodes, but we need to pass maxnode+1

         * as argument to mbind() due to an old Linux bug (feature?) which

         * cuts off the last specified node. This means backend->host_nodes

         * must have MAX_NODES+1 bits available.

         */

        assert(sizeof(backend->host_nodes) >=

               BITS_TO_LONGS(MAX_NODES + 1) * sizeof(unsigned long));

        assert(maxnode <= MAX_NODES);

        if (mbind(ptr, sz, backend->policy,

                  maxnode ? backend->host_nodes : NULL, maxnode + 1, flags)) {

            if (backend->policy != MPOL_DEFAULT || errno != ENOSYS) {

                error_setg_errno(errp, errno,

                                 "cannot bind memory to host NUMA nodes");

                return;

            }

        }

#endif

        /* Preallocate memory after the NUMA policy has been instantiated.

         * This is necessary to guarantee memory is allocated with

         * specified NUMA policy in place.

         */

        if (backend->prealloc) {

            os_mem_prealloc(memory_region_get_fd(&backend->mr), ptr, sz,

                            &local_err);

            if (local_err) {

                goto out;

            }

        }

    }

out:

    error_propagate(errp, local_err);

}
