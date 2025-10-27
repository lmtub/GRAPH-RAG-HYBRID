char_socket_get_addr(Object *obj, Visitor *v, const char *name,

                     void *opaque, Error **errp)

{

    SocketChardev *s = SOCKET_CHARDEV(obj);



    visit_type_SocketAddress(v, name, &s->addr, errp);

}
