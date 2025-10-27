static void coroutine_fn nest(void *opaque)

{

    NestData *nd = opaque;



    nd->n_enter++;



    if (nd->n_enter < nd->max) {

        Coroutine *child;



        child = qemu_coroutine_create(nest);

        qemu_coroutine_enter(child, nd);

    }



    nd->n_return++;

}
