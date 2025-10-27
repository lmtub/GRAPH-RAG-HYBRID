void async_context_push(void)

{

    struct AsyncContext *new = qemu_mallocz(sizeof(*new));

    new->parent = async_context;

    new->id = async_context->id + 1;

    async_context = new;

}
