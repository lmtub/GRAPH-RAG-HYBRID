void run_on_cpu(CPUState *env, void (*func)(void *data), void *data)

{

    func(data);

}
