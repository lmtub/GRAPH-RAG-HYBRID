void coroutine_fn qemu_coroutine_yield(void)

{

    Coroutine *self = qemu_coroutine_self();

    Coroutine *to = self->caller;



    trace_qemu_coroutine_yield(self, to);



    if (!to) {

        fprintf(stderr, "Co-routine is yielding to no one\n");

        abort();

    }



    self->caller = NULL;

    coroutine_swap(self, to);

}
