#ifndef GUARD_ccv_nnc_stream_fwd_h
#define GUARD_ccv_nnc_stream_fwd_h

typedef struct co_routine_s co_routine_t;

typedef struct {
	ccv_nnc_callback_f fn;
	void* callback_context;
} ccv_nnc_async_callback_t;

typedef void(*ccv_nnc_async_callback_f)(ccv_nnc_async_callback_t* const async);

#endif
