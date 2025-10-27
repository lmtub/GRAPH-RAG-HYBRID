int qemu_add_wait_object(HANDLE handle, WaitObjectFunc *func, void *opaque)

{

    WaitObjects *w = &wait_objects;



    if (w->num >= MAXIMUM_WAIT_OBJECTS)

        return -1;

    w->events[w->num] = handle;

    w->func[w->num] = func;

    w->opaque[w->num] = opaque;

    w->num++;

    return 0;

}
