/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-09T23:04:19+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "lite_operators.h"

#include "ai_lite_inspect.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x314106caaa2bea05e790422ca2746c6b"
#define _STAI_NETWORK_DATETIME            "2026-03-09T23:04:19+0100"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 0),
  .tool_version = STAI_INIT_VERSION(4, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 97, 40, 1),
      STAI_DECLARE_ARRAY(float, 1, 0.08156336843967438f),
      STAI_DECLARE_ARRAY(int16_t, 1, -10)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 2),
      STAI_DECLARE_ARRAY(float, 1, 0.00390625f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 342340),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 33160),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09375828504562378f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_2_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010783391073346138f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00901670940220356f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_3_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002209884114563465f, 0.0024065617471933365f, 0.002392158843576908f, 0.0025342488661408424f, 0.0021531046368181705f, 0.0021949810907244682f, 0.0021461539436131716f, 0.0022881426848471165f, 0.0022006193175911903f, 0.0024971626698970795f, 0.002258794382214546f, 0.002497157547622919f, 0.002155896043404937f, 0.00228649890050292f, 0.002434961963444948f, 0.0024288631975650787f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013431535102427006f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_4_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002096593612805009f, 0.0032816645689308643f, 0.0023283720947802067f, 0.0023052794858813286f, 0.002036527032032609f, 0.0022586158011108637f, 0.0024670432321727276f, 0.00170594931114465f, 0.002371198730543256f, 0.00232772552408278f, 0.0023534391075372696f, 0.0023428392596542835f, 0.002476163674145937f, 0.002038300037384033f, 0.002146380254998803f, 0.002216269727796316f, 0.0021675359457731247f, 0.0022689371835440397f, 0.002442616969347f, 0.0022178080398589373f, 0.002064688829705119f, 0.0025750792119652033f, 0.0026078964583575726f, 0.0023478553630411625f, 0.0019852500408887863f, 0.002728799358010292f, 0.002024132991209626f, 0.002117194002494216f, 0.0023589786142110825f, 0.002643812447786331f, 0.002068611327558756f, 0.002474926644936204f, 0.0021786768920719624f, 0.0017997754039242864f, 0.002378119621425867f, 0.0014798827469348907f, 0.0024014085065573454f, 0.002779558766633272f, 0.0024655202869325876f, 0.0020065405406057835f, 0.002184811048209667f, 0.0021993531845510006f, 0.0026084971614181995f, 0.0025912595447152853f, 0.0026367707177996635f, 0.0020524414721876383f, 0.0018738758517429233f, 0.0022036065347492695f, 0.002083639381453395f, 0.0021401559934020042f, 0.0023671414237469435f, 0.002496077911928296f, 0.002062418730929494f, 0.0022662377450615168f, 0.0020348690450191498f, 0.0020736660808324814f, 0.0022502150386571884f, 0.0017154846573248506f, 0.0021163804922252893f, 0.002217358909547329f, 0.002318017650395632f, 0.0019345908658578992f, 0.0025273736100643873f, 0.0020706302020698786f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_10_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.055396609008312225f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_12_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12319736927747726f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_13_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0045121414586901665f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007730517536401749f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0024371931795030832f, 0.002437667455524206f, 0.0021297121420502663f, 0.002424640581011772f, 0.002553581027314067f, 0.0022443411871790886f, 0.0020157666876912117f, 0.0023369959089905024f, 0.002547190058976412f, 0.0021179048344492912f, 0.0021378370001912117f, 0.0026709360536187887f, 0.0022751064971089363f, 0.0023379840422421694f, 0.0022440289612859488f, 0.002231672639027238f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_15_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00851499568670988f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_15_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0022428010124713182f, 0.002410800429061055f, 0.0018513620598241687f, 0.0027057810220867395f, 0.0022386310156434774f, 0.002138750394806266f, 0.0021107785869389772f, 0.0022276134695857763f, 0.0023062322288751602f, 0.002123839221894741f, 0.002442789264023304f, 0.001902862684801221f, 0.0024855174124240875f, 0.002121871802955866f, 0.0018519095610827208f, 0.0021454247180372477f, 0.002175390487536788f, 0.002239611465483904f, 0.002353796735405922f, 0.0021073936950415373f, 0.0023390569258481264f, 0.002910604700446129f, 0.002164390403777361f, 0.0021379035897552967f, 0.0021619643084704876f, 0.002251512138172984f, 0.0021036313846707344f, 0.0020418623462319374f, 0.002118881093338132f, 0.002040569670498371f, 0.002188638085499406f, 0.002114840317517519f, 0.0018507493659853935f, 0.0018193669384345412f, 0.002429964719340205f, 0.0022488341201096773f, 0.0021384034771472216f, 0.002051087561994791f, 0.0023754010908305645f, 0.0021475241519510746f, 0.0019309764029458165f, 0.0019994820468127728f, 0.0023690960370004177f, 0.0023144076112657785f, 0.0017848694697022438f, 0.002195715671405196f, 0.002228202298283577f, 0.0019518976332619786f, 0.0019791305530816317f, 0.0020522079430520535f, 0.0020660331938415766f, 0.0019203175324946642f, 0.0020494614727795124f, 0.0019403001060709357f, 0.0022323194425553083f, 0.0028699033427983522f, 0.0019493342842906713f, 0.0024002802092581987f, 0.0018983428599312901f, 0.0024810011964291334f, 0.0018898188136518002f, 0.002178326016291976f, 0.0018637394532561302f, 0.0022904928773641586f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_16_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_21_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05964299291372299f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_23_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12380179762840271f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_24_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006248328369110823f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_25_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013639613054692745f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_25_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0031191317830234766f, 0.0023787508253008127f, 0.002710524247959256f, 0.002069018082693219f, 0.002298821462318301f, 0.0021625529043376446f, 0.0031795832328498363f, 0.003110474906861782f, 0.0025574828032404184f, 0.0023663805332034826f, 0.003092752071097493f, 0.0026733402628451586f, 0.0023487936705350876f, 0.0028731145430356264f, 0.0031912808772176504f, 0.0023190579377114773f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_26_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015937188640236855f),
    AI_PACK_INTQ_ZP(38)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_26_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0030599397141486406f, 0.00212275143712759f, 0.0022665862925350666f, 0.0027643844950944185f, 0.001992074539884925f, 0.002093078102916479f, 0.002796816872432828f, 0.0025562315713614225f, 0.0020780558697879314f, 0.0027112250681966543f, 0.0021323300898075104f, 0.002070529153570533f, 0.002661663806065917f, 0.0022403455805033445f, 0.0031767648179084063f, 0.002284521237015724f, 0.0026864251121878624f, 0.0019346154294908047f, 0.003038457129150629f, 0.002836810424923897f, 0.0031630934681743383f, 0.0025955981109291315f, 0.0016243329737335443f, 0.002336160046979785f, 0.0031758679542690516f, 0.002680001547560096f, 0.0022560858633369207f, 0.0019734755624085665f, 0.0023871054872870445f, 0.0022527077235281467f, 0.002767392201349139f, 0.0023110953625291586f, 0.002515156287699938f, 0.0026974263601005077f, 0.0020275216083973646f, 0.0028034416027367115f, 0.002233336679637432f, 0.0021000502165406942f, 0.002697618678212166f, 0.002169556450098753f, 0.0020306194201111794f, 0.0024782021064311266f, 0.0027516523841768503f, 0.0024047791957855225f, 0.0023279048036783934f, 0.0026875718031078577f, 0.0018688406562432647f, 0.002998441457748413f, 0.002225998556241393f, 0.0025412419345229864f, 0.0021715222392231226f, 0.002630322938784957f, 0.0019654000643640757f, 0.0026880858931690454f, 0.0021906839683651924f, 0.0019609988667070866f, 0.002138826996088028f, 0.002146813552826643f, 0.0029420312494039536f, 0.003494350705295801f, 0.0027291737496852875f, 0.0022549571003764868f, 0.002081074984744191f, 0.002217907225713134f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_27_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_32_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08252226561307907f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_34_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12345076352357864f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_35_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006445815321058035f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_36_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.014802264049649239f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_36_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0036137443967163563f, 0.005014938767999411f, 0.0027445105370134115f, 0.00350546813569963f, 0.0029009459540247917f, 0.0046991268172860146f, 0.004855580627918243f, 0.005662869196385145f, 0.002177711110562086f, 0.003400244517251849f, 0.0022181763779371977f, 0.0021636884193867445f, 0.0024184761568903923f, 0.0026377749163657427f, 0.003749985247850418f, 0.0026540705002844334f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_37_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02363372966647148f),
    AI_PACK_INTQ_ZP(40)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_37_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003967718221247196f, 0.0023713752161711454f, 0.002949804998934269f, 0.0030687132384628057f, 0.002794342115521431f, 0.0037371960934251547f, 0.004934856668114662f, 0.003341825446113944f, 0.0035801425110548735f, 0.003779004095122218f, 0.0036151367239654064f, 0.002128197345882654f, 0.0020396027248352766f, 0.0027543383184820414f, 0.0032013882882893085f, 0.0036425369326025248f, 0.003112934762611985f, 0.004608611576259136f, 0.002614561701193452f, 0.002417834708467126f, 0.002856323728337884f, 0.002455789130181074f, 0.0034723689313977957f, 0.0018247487023472786f, 0.0034916687291115522f, 0.0026238402351737022f, 0.003336624475196004f, 0.002250503282994032f, 0.0019285492599010468f, 0.0032477921340614557f, 0.0031478628516197205f, 0.0036102873273193836f, 0.0020974769722670317f, 0.0044220914132893085f, 0.004588539712131023f, 0.003199780359864235f, 0.0038513168692588806f, 0.0020861870143562555f, 0.0025761090219020844f, 0.004073195159435272f, 0.002564758062362671f, 0.0031514069996774197f, 0.0027879108674824238f, 0.003044347045943141f, 0.0035558678209781647f, 0.00267984950914979f, 0.0032546480651944876f, 0.004164029378443956f, 0.0035589290782809258f, 0.0030840567778795958f, 0.004146440885961056f, 0.0031803969759494066f, 0.002292752731591463f, 0.003018118441104889f, 0.0028149252757430077f, 0.002966745523735881f, 0.002421513432636857f, 0.0025578171480447054f, 0.002596083329990506f, 0.004691864363849163f, 0.004026615992188454f, 0.004387902561575174f, 0.0031241143587976694f, 0.0036039522383362055f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_38_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_43_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04626816138625145f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_44_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11792958527803421f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_45_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009350219741463661f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_46_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08059944957494736f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_46_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 2,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003036095527932048f, 0.0029616551473736763f),
    AI_PACK_INTQ_ZP(0, 0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 248320, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  pool_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  gemm_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  gemm_3_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  gemm_3_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gemm_3_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 336, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  nl_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 248320, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 248320, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  pool_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  gemm_15_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 336, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  nl_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 248320, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_23_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 248320, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  pool_24_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  gemm_25_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  gemm_25_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  gemm_25_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  gemm_25_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  gemm_26_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 336, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  nl_27_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 248320, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_34_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 248320, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  pool_35_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  gemm_36_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 144, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  gemm_37_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  gemm_37_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  gemm_37_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  gemm_37_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 336, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  nl_38_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_43_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 248320, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_44_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 248320, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  pool_45_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 2, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  gemm_46_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 74, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 97), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  pool_2_output, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_2_output_array, &pool_2_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  gemm_3_bias, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gemm_3_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  gemm_3_output, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_3_output_array, &gemm_3_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  gemm_3_scratch0, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_3_scratch0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  gemm_3_weights, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 64, 16, 1, 1), AI_STRIDE_INIT(4, 1, 64, 1024, 1024),
  1, &gemm_3_weights_array, &gemm_3_weights_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_bias, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_4_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_output, AI_STATIC,
  77, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_4_output_array, &gemm_4_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_scratch0, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 2, 2, 672, 672),
  1, &gemm_4_scratch0_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_weights, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 16, 64, 1, 1), AI_STRIDE_INIT(4, 1, 16, 1024, 1024),
  1, &gemm_4_weights_array, &gemm_4_weights_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  nl_5_output, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_5_output_array, &nl_5_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_10_output, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 97), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &eltwise_10_output_array, &eltwise_10_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 97), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &conv2d_12_output_array, &conv2d_12_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  pool_13_output, AI_STATIC,
  86, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_13_output_array, &pool_13_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_bias, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gemm_14_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_14_output_array, &gemm_14_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_scratch0, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_14_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_weights, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 64, 16, 1, 1), AI_STRIDE_INIT(4, 1, 64, 1024, 1024),
  1, &gemm_14_weights_array, &gemm_14_weights_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_bias, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_15_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_15_output_array, &gemm_15_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_scratch0, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 2, 2, 672, 672),
  1, &gemm_15_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  gemm_15_weights, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 16, 64, 1, 1), AI_STRIDE_INIT(4, 1, 16, 1024, 1024),
  1, &gemm_15_weights_array, &gemm_15_weights_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  nl_16_output, AI_STATIC,
  80, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_16_output_array, &nl_16_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_21_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 97), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &eltwise_21_output_array, &eltwise_21_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_23_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 97), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &conv2d_23_output_array, &conv2d_23_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  pool_24_output, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_24_output_array, &pool_24_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  gemm_25_bias, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gemm_25_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  gemm_25_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_25_output_array, &gemm_25_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  gemm_25_scratch0, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_25_scratch0_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  gemm_25_weights, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 64, 16, 1, 1), AI_STRIDE_INIT(4, 1, 64, 1024, 1024),
  1, &gemm_25_weights_array, &gemm_25_weights_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_bias, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_26_bias_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_26_output_array, &gemm_26_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_scratch0, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 2, 2, 672, 672),
  1, &gemm_26_scratch0_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  gemm_26_weights, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 16, 64, 1, 1), AI_STRIDE_INIT(4, 1, 16, 1024, 1024),
  1, &gemm_26_weights_array, &gemm_26_weights_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  nl_27_output, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_27_output_array, &nl_27_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_32_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 97), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &eltwise_32_output_array, &eltwise_32_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_34_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 97), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &conv2d_34_output_array, &conv2d_34_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  pool_35_output, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_35_output_array, &pool_35_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_bias, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gemm_36_bias_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &gemm_36_output_array, &gemm_36_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_scratch0, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 2, 2, 288, 288),
  1, &gemm_36_scratch0_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  gemm_36_weights, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(4, 64, 16, 1, 1), AI_STRIDE_INIT(4, 1, 64, 1024, 1024),
  1, &gemm_36_weights_array, &gemm_36_weights_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  gemm_37_bias, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_37_bias_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  gemm_37_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_37_output_array, &gemm_37_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  gemm_37_scratch0, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 2, 2, 672, 672),
  1, &gemm_37_scratch0_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  gemm_37_weights, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 16, 64, 1, 1), AI_STRIDE_INIT(4, 1, 16, 1024, 1024),
  1, &gemm_37_weights_array, &gemm_37_weights_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  nl_38_output, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_38_output_array, &nl_38_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_43_output, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 97), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &eltwise_43_output_array, &eltwise_43_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_44_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 97), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &conv2d_44_output_array, &conv2d_44_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  pool_45_output, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_45_output_array, &pool_45_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_bias, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &gemm_46_bias_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_output, AI_STATIC,
  73, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2, 2),
  1, &gemm_46_output_array, &gemm_46_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_scratch0, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 74, 1, 1), AI_STRIDE_INIT(4, 2, 2, 148, 148),
  1, &gemm_46_scratch0_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  gemm_46_weights, AI_STATIC,
  75, 0x1,
  AI_SHAPE_INIT(4, 64, 2, 1, 1), AI_STRIDE_INIT(4, 1, 64, 128, 128),
  1, &gemm_46_weights_array, &gemm_46_weights_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_2_layer, 2,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_2_chain,
  NULL, &pool_2_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(40, 97), 
  .pool_stride = AI_SHAPE_2D_INIT(40, 97), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_3_weights, &gemm_3_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_3_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_3_layer, 3,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_3_chain,
  NULL, &gemm_3_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_4_weights, &gemm_4_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_4_layer, 4,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_4_chain,
  NULL, &gemm_4_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_5_nl_params_data[] = { -83, -83, -82, -82, -81, -81, -80, -80, -79, -79, -78, -78, -77, -77, -76, -76, -75, -74, -74, -73, -73, -72, -72, -71, -70, -70, -69, -69, -68, -67, -67, -66, -65, -65, -64, -63, -63, -62, -62, -61, -60, -60, -59, -58, -57, -57, -56, -55, -55, -54, -53, -53, -52, -51, -50, -50, -49, -48, -47, -47, -46, -45, -44, -44, -43, -42, -41, -41, -40, -39, -38, -38, -37, -36, -35, -34, -34, -33, -32, -31, -30, -30, -29, -28, -27, -26, -25, -25, -24, -23, -22, -21, -20, -20, -19, -18, -17, -16, -15, -15, -14, -13, -12, -11, -10, -9, -9, -8, -7, -6, -5, -4, -3, -3, -2, -1, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 25, 26, 27, 28, 29, 30, 30, 31, 32, 33, 34, 34, 35, 36, 37, 38, 38, 39, 40, 41, 41, 42, 43, 44, 44, 45, 46, 47, 47, 48, 49, 50, 50, 51, 52, 53, 53, 54, 55, 55, 56, 57, 57, 58, 59, 60, 60, 61, 62, 62, 63, 63, 64, 65, 65, 66, 67, 67, 68, 69, 69, 70, 70, 71, 72, 72, 73, 73, 74, 74, 75, 76, 76, 77, 77, 78, 78, 79, 79, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90, 90, 90, 91, 91, 92, 92, 93, 93, 93, 94 };
AI_ARRAY_OBJ_DECLARE(
    nl_5_nl_params, AI_ARRAY_FORMAT_S8,
    nl_5_nl_params_data, nl_5_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_5_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_5_chain,
  NULL, &nl_5_layer, AI_STATIC, 
  .nl_params = &nl_5_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_1_output, &nl_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_10_layer, 10,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_10_chain,
  NULL, &eltwise_10_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_13_layer, 13,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_13_chain,
  NULL, &pool_13_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(40, 97), 
  .pool_stride = AI_SHAPE_2D_INIT(40, 97), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_14_weights, &gemm_14_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_14_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_14_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_15_weights, &gemm_15_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_15_layer, 15,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_15_chain,
  NULL, &gemm_15_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_16_nl_params_data[] = { -73, -72, -72, -71, -71, -71, -70, -70, -70, -69, -69, -68, -68, -68, -67, -67, -66, -66, -66, -65, -65, -64, -64, -64, -63, -63, -62, -62, -62, -61, -61, -60, -60, -59, -59, -59, -58, -58, -57, -57, -56, -56, -55, -55, -55, -54, -54, -53, -53, -52, -52, -51, -51, -51, -50, -50, -49, -49, -48, -48, -47, -47, -46, -46, -45, -45, -44, -44, -43, -43, -42, -42, -41, -41, -41, -40, -40, -39, -39, -38, -38, -37, -37, -36, -36, -35, -35, -34, -34, -33, -33, -32, -31, -31, -30, -30, -29, -29, -28, -28, -27, -27, -26, -26, -25, -25, -24, -24, -23, -23, -22, -22, -21, -21, -20, -19, -19, -18, -18, -17, -17, -16, -16, -15, -15, -14, -14, -13, -12, -12, -11, -11, -10, -10, -9, -9, -8, -8, -7, -7, -6, -5, -5, -4, -4, -3, -3, -2, -2, -1, -1, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 51, 52, 52, 53, 53 };
AI_ARRAY_OBJ_DECLARE(
    nl_16_nl_params, AI_ARRAY_FORMAT_S8,
    nl_16_nl_params_data, nl_16_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_16_layer, 16,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_16_chain,
  NULL, &nl_16_layer, AI_STATIC, 
  .nl_params = &nl_16_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_12_output, &nl_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_21_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_21_layer, 21,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_21_chain,
  NULL, &eltwise_21_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_24_layer, 24,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_24_chain,
  NULL, &pool_24_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(40, 97), 
  .pool_stride = AI_SHAPE_2D_INIT(40, 97), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_25_weights, &gemm_25_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_25_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_25_layer, 25,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_25_chain,
  NULL, &gemm_25_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_26_weights, &gemm_26_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_26_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_26_layer, 26,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_26_chain,
  NULL, &gemm_26_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_27_nl_params_data[] = { -111, -111, -111, -110, -110, -110, -109, -109, -109, -109, -108, -108, -108, -107, -107, -107, -107, -106, -106, -106, -105, -105, -105, -104, -104, -104, -103, -103, -102, -102, -102, -101, -101, -101, -100, -100, -99, -99, -99, -98, -98, -97, -97, -96, -96, -96, -95, -95, -94, -94, -93, -93, -92, -92, -91, -91, -90, -90, -89, -89, -88, -88, -87, -86, -86, -85, -85, -84, -84, -83, -82, -82, -81, -81, -80, -79, -79, -78, -77, -77, -76, -75, -75, -74, -73, -73, -72, -71, -71, -70, -69, -69, -68, -67, -66, -66, -65, -64, -63, -63, -62, -61, -60, -59, -59, -58, -57, -56, -55, -54, -54, -53, -52, -51, -50, -49, -48, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51, 52, 53, 54, 54, 55, 56, 57, 58, 59, 59, 60, 61, 62, 63, 63, 64, 65, 66, 66, 67, 68, 69, 69, 70, 71, 71, 72, 73, 73, 74, 75, 75, 76, 77, 77, 78 };
AI_ARRAY_OBJ_DECLARE(
    nl_27_nl_params, AI_ARRAY_FORMAT_S8,
    nl_27_nl_params_data, nl_27_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_27_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_27_layer, 27,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_27_chain,
  NULL, &nl_27_layer, AI_STATIC, 
  .nl_params = &nl_27_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_23_output, &nl_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_32_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_32_layer, 32,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_32_chain,
  NULL, &eltwise_32_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_35_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_35_layer, 35,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_35_chain,
  NULL, &pool_35_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(40, 97), 
  .pool_stride = AI_SHAPE_2D_INIT(40, 97), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_36_weights, &gemm_36_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_36_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_36_layer, 36,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_36_chain,
  NULL, &gemm_36_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_37_weights, &gemm_37_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_37_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_37_layer, 37,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_37_chain,
  NULL, &gemm_37_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_38_nl_params_data[] = { -123, -123, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -117, -117, -117, -117, -116, -116, -116, -116, -115, -115, -115, -114, -114, -114, -113, -113, -113, -112, -112, -112, -111, -111, -111, -110, -110, -110, -109, -109, -108, -108, -107, -107, -106, -106, -106, -105, -105, -104, -103, -103, -102, -102, -101, -101, -100, -100, -99, -98, -98, -97, -96, -96, -95, -94, -94, -93, -92, -92, -91, -90, -89, -89, -88, -87, -86, -85, -84, -84, -83, -82, -81, -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -63, -62, -61, -60, -59, -58, -56, -55, -54, -53, -51, -50, -49, -48, -46, -45, -44, -42, -41, -40, -38, -37, -35, -34, -33, -31, -30, -28, -27, -25, -24, -22, -21, -20, -18, -17, -15, -14, -12, -11, -9, -8, -6, -5, -3, -2, 0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 35, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 84, 85, 86, 87, 88, 89, 89, 90, 91, 92, 92, 93, 94, 94, 95, 96, 96, 97, 98, 98, 99 };
AI_ARRAY_OBJ_DECLARE(
    nl_38_nl_params, AI_ARRAY_FORMAT_S8,
    nl_38_nl_params_data, nl_38_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_38_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_38_layer, 38,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_38_chain,
  NULL, &nl_38_layer, AI_STATIC, 
  .nl_params = &nl_38_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_34_output, &nl_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_43_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_43_layer, 43,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_43_chain,
  NULL, &eltwise_43_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_44_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_45_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_45_layer, 45,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_45_chain,
  NULL, &pool_45_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(40, 97), 
  .pool_stride = AI_SHAPE_2D_INIT(40, 97), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_46_weights, &gemm_46_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_46_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_46_layer, 46,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_46_chain,
  NULL, &gemm_46_layer, AI_STATIC, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_ap_integer_INT8_pool_2(_stai_network_context* net_ctx)
{
  conv2d_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 71296);
  conv2d_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 71296);
  pool_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 319616);
  pool_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 319616);
  _STAI_NETWORK_EVENT_NODE_START_CB(2, 1, { conv2d_1_output.data->data});
  forward_ap_integer_INT8(&pool_2_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(2, 1, { pool_2_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_3(_stai_network_context* net_ctx)
{
  pool_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 319616);
  pool_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 319616);
  gemm_3_weights_array.data = AI_PTR(net_ctx->_weights[0] + 3648);
  gemm_3_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 3648);
  gemm_3_bias_array.data = AI_PTR(net_ctx->_weights[0] + 4672);
  gemm_3_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 4672);
  gemm_3_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 319680);
  gemm_3_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 319680);
  gemm_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 319968);
  gemm_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 319968);
  _STAI_NETWORK_EVENT_NODE_START_CB(3, 1, { pool_2_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(3, 1, { gemm_3_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_4(_stai_network_context* net_ctx)
{
  gemm_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 319968);
  gemm_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 319968);
  gemm_4_weights_array.data = AI_PTR(net_ctx->_weights[0] + 4736);
  gemm_4_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 4736);
  gemm_4_bias_array.data = AI_PTR(net_ctx->_weights[0] + 5760);
  gemm_4_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 5760);
  gemm_4_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 319984);
  gemm_4_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 319984);
  gemm_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 319616);
  gemm_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 319616);
  _STAI_NETWORK_EVENT_NODE_START_CB(4, 1, { gemm_3_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(4, 1, { gemm_4_output.data->data});
}
void forward_lite_nl_integer_nl_5(_stai_network_context* net_ctx)
{
  gemm_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 319616);
  gemm_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 319616);
  nl_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 319680);
  nl_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 319680);
  _STAI_NETWORK_EVENT_NODE_START_CB(5, 1, { gemm_4_output.data->data});
  forward_nl_integer(&nl_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(5, 1, { nl_5_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_10(_stai_network_context* net_ctx)
{
  conv2d_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 71296);
  conv2d_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 71296);
  nl_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 319680);
  nl_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 319680);
  eltwise_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 71296);
  eltwise_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 71296);
  _STAI_NETWORK_EVENT_NODE_START_CB(10, 2, { conv2d_1_output.data->data,nl_5_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_10_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(10, 1, { eltwise_10_output.data->data});
}
void forward_lite_ap_integer_INT8_pool_13(_stai_network_context* net_ctx)
{
  conv2d_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 48384);
  conv2d_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 48384);
  pool_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 296704);
  pool_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 296704);
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 1, { conv2d_12_output.data->data});
  forward_ap_integer_INT8(&pool_13_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, { pool_13_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_14(_stai_network_context* net_ctx)
{
  pool_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 296704);
  pool_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 296704);
  gemm_14_weights_array.data = AI_PTR(net_ctx->_weights[0] + 11200);
  gemm_14_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 11200);
  gemm_14_bias_array.data = AI_PTR(net_ctx->_weights[0] + 12224);
  gemm_14_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 12224);
  gemm_14_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 296768);
  gemm_14_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 296768);
  gemm_14_output_array.data = AI_PTR(net_ctx->_activations[0] + 297056);
  gemm_14_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 297056);
  _STAI_NETWORK_EVENT_NODE_START_CB(14, 1, { pool_13_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_14_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(14, 1, { gemm_14_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_15(_stai_network_context* net_ctx)
{
  gemm_14_output_array.data = AI_PTR(net_ctx->_activations[0] + 297056);
  gemm_14_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 297056);
  gemm_15_weights_array.data = AI_PTR(net_ctx->_weights[0] + 12288);
  gemm_15_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 12288);
  gemm_15_bias_array.data = AI_PTR(net_ctx->_weights[0] + 13312);
  gemm_15_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 13312);
  gemm_15_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 297072);
  gemm_15_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 297072);
  gemm_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 296704);
  gemm_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 296704);
  _STAI_NETWORK_EVENT_NODE_START_CB(15, 1, { gemm_14_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_15_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(15, 1, { gemm_15_output.data->data});
}
void forward_lite_nl_integer_nl_16(_stai_network_context* net_ctx)
{
  gemm_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 296704);
  gemm_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 296704);
  nl_16_output_array.data = AI_PTR(net_ctx->_activations[0] + 296768);
  nl_16_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 296768);
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 1, { gemm_15_output.data->data});
  forward_nl_integer(&nl_16_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, { nl_16_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_21(_stai_network_context* net_ctx)
{
  conv2d_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 48384);
  conv2d_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 48384);
  nl_16_output_array.data = AI_PTR(net_ctx->_activations[0] + 296768);
  nl_16_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 296768);
  eltwise_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 48384);
  eltwise_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 48384);
  _STAI_NETWORK_EVENT_NODE_START_CB(21, 2, { conv2d_12_output.data->data,nl_16_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_21_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(21, 1, { eltwise_21_output.data->data});
}
void forward_lite_ap_integer_INT8_pool_24(_stai_network_context* net_ctx)
{
  conv2d_23_output_array.data = AI_PTR(net_ctx->_activations[0] + 25472);
  conv2d_23_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 25472);
  pool_24_output_array.data = AI_PTR(net_ctx->_activations[0] + 273792);
  pool_24_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 273792);
  _STAI_NETWORK_EVENT_NODE_START_CB(24, 1, { conv2d_23_output.data->data});
  forward_ap_integer_INT8(&pool_24_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(24, 1, { pool_24_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_25(_stai_network_context* net_ctx)
{
  pool_24_output_array.data = AI_PTR(net_ctx->_activations[0] + 273792);
  pool_24_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 273792);
  gemm_25_weights_array.data = AI_PTR(net_ctx->_weights[0] + 18752);
  gemm_25_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 18752);
  gemm_25_bias_array.data = AI_PTR(net_ctx->_weights[0] + 19776);
  gemm_25_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 19776);
  gemm_25_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 273856);
  gemm_25_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 273856);
  gemm_25_output_array.data = AI_PTR(net_ctx->_activations[0] + 274144);
  gemm_25_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 274144);
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, { pool_24_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_25_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, { gemm_25_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_26(_stai_network_context* net_ctx)
{
  gemm_25_output_array.data = AI_PTR(net_ctx->_activations[0] + 274144);
  gemm_25_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 274144);
  gemm_26_weights_array.data = AI_PTR(net_ctx->_weights[0] + 19840);
  gemm_26_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 19840);
  gemm_26_bias_array.data = AI_PTR(net_ctx->_weights[0] + 20864);
  gemm_26_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 20864);
  gemm_26_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 274160);
  gemm_26_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 274160);
  gemm_26_output_array.data = AI_PTR(net_ctx->_activations[0] + 273792);
  gemm_26_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 273792);
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 1, { gemm_25_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_26_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, { gemm_26_output.data->data});
}
void forward_lite_nl_integer_nl_27(_stai_network_context* net_ctx)
{
  gemm_26_output_array.data = AI_PTR(net_ctx->_activations[0] + 273792);
  gemm_26_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 273792);
  nl_27_output_array.data = AI_PTR(net_ctx->_activations[0] + 273856);
  nl_27_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 273856);
  _STAI_NETWORK_EVENT_NODE_START_CB(27, 1, { gemm_26_output.data->data});
  forward_nl_integer(&nl_27_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(27, 1, { nl_27_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_32(_stai_network_context* net_ctx)
{
  conv2d_23_output_array.data = AI_PTR(net_ctx->_activations[0] + 25472);
  conv2d_23_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 25472);
  nl_27_output_array.data = AI_PTR(net_ctx->_activations[0] + 273856);
  nl_27_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 273856);
  eltwise_32_output_array.data = AI_PTR(net_ctx->_activations[0] + 25472);
  eltwise_32_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 25472);
  _STAI_NETWORK_EVENT_NODE_START_CB(32, 2, { conv2d_23_output.data->data,nl_27_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_32_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(32, 1, { eltwise_32_output.data->data});
}
void forward_lite_ap_integer_INT8_pool_35(_stai_network_context* net_ctx)
{
  conv2d_34_output_array.data = AI_PTR(net_ctx->_activations[0] + 2560);
  conv2d_34_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2560);
  pool_35_output_array.data = AI_PTR(net_ctx->_activations[0] + 250880);
  pool_35_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 250880);
  _STAI_NETWORK_EVENT_NODE_START_CB(35, 1, { conv2d_34_output.data->data});
  forward_ap_integer_INT8(&pool_35_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(35, 1, { pool_35_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_36(_stai_network_context* net_ctx)
{
  pool_35_output_array.data = AI_PTR(net_ctx->_activations[0] + 250880);
  pool_35_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 250880);
  gemm_36_weights_array.data = AI_PTR(net_ctx->_weights[0] + 26304);
  gemm_36_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 26304);
  gemm_36_bias_array.data = AI_PTR(net_ctx->_weights[0] + 27328);
  gemm_36_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 27328);
  gemm_36_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 250944);
  gemm_36_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 250944);
  gemm_36_output_array.data = AI_PTR(net_ctx->_activations[0] + 251232);
  gemm_36_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 251232);
  _STAI_NETWORK_EVENT_NODE_START_CB(36, 1, { pool_35_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_36_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(36, 1, { gemm_36_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_37(_stai_network_context* net_ctx)
{
  gemm_36_output_array.data = AI_PTR(net_ctx->_activations[0] + 251232);
  gemm_36_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 251232);
  gemm_37_weights_array.data = AI_PTR(net_ctx->_weights[0] + 27392);
  gemm_37_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 27392);
  gemm_37_bias_array.data = AI_PTR(net_ctx->_weights[0] + 28416);
  gemm_37_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 28416);
  gemm_37_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 251248);
  gemm_37_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 251248);
  gemm_37_output_array.data = AI_PTR(net_ctx->_activations[0] + 250880);
  gemm_37_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 250880);
  _STAI_NETWORK_EVENT_NODE_START_CB(37, 1, { gemm_36_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_37_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(37, 1, { gemm_37_output.data->data});
}
void forward_lite_nl_integer_nl_38(_stai_network_context* net_ctx)
{
  gemm_37_output_array.data = AI_PTR(net_ctx->_activations[0] + 250880);
  gemm_37_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 250880);
  nl_38_output_array.data = AI_PTR(net_ctx->_activations[0] + 250944);
  nl_38_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 250944);
  _STAI_NETWORK_EVENT_NODE_START_CB(38, 1, { gemm_37_output.data->data});
  forward_nl_integer(&nl_38_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(38, 1, { nl_38_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_43(_stai_network_context* net_ctx)
{
  conv2d_34_output_array.data = AI_PTR(net_ctx->_activations[0] + 2560);
  conv2d_34_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2560);
  nl_38_output_array.data = AI_PTR(net_ctx->_activations[0] + 250944);
  nl_38_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 250944);
  eltwise_43_output_array.data = AI_PTR(net_ctx->_activations[0] + 2560);
  eltwise_43_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2560);
  _STAI_NETWORK_EVENT_NODE_START_CB(43, 2, { conv2d_34_output.data->data,nl_38_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_43_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(43, 1, { eltwise_43_output.data->data});
}
void forward_lite_ap_integer_INT8_pool_45(_stai_network_context* net_ctx)
{
  conv2d_44_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_44_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  pool_45_output_array.data = AI_PTR(net_ctx->_activations[0] + 248320);
  pool_45_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 248320);
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, { conv2d_44_output.data->data});
  forward_ap_integer_INT8(&pool_45_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, { pool_45_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_46(_stai_network_context* net_ctx)
{
  pool_45_output_array.data = AI_PTR(net_ctx->_activations[0] + 248320);
  pool_45_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 248320);
  gemm_46_weights_array.data = AI_PTR(net_ctx->_weights[0] + 33024);
  gemm_46_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 33024);
  gemm_46_bias_array.data = AI_PTR(net_ctx->_weights[0] + 33152);
  gemm_46_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 33152);
  gemm_46_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  gemm_46_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  gemm_46_output_array.data = AI_PTR(net_ctx->_activations[0] + 148);
  gemm_46_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 148);
  _STAI_NETWORK_EVENT_NODE_START_CB(46, 1, { pool_45_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_46_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(46, 1, { gemm_46_output.data->data});
}

/*****************************************************************************/


static const ai_u16 conv2d_0_t_in_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_0_t_in_0_shape_h_const_u16 = 97;
static const ai_u16 conv2d_0_t_in_0_shape_ch_const_u16 = 1;
static const ai_u16 conv2d_0_t_out_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_0_t_weight_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_0_t_weight_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_0_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_0_l_stride_0_const_u16 = 1;
static const ai_i32 conv2d_0_l_pad_W_0_const_s32 = 4;
static const ai_i32 conv2d_0_l_pad_H_0_const_s32 = 1;
static const ai_i8 conv2d_0_t_in_0_fmt_zero_const_s8 = -10;
static const ai_i8 conv2d_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_0_t_in_0_fmt_scale_const_f32 = 0.08156336843967438f;
static const ai_float conv2d_0_t_out_0_fmt_scale_const_f32 = 0.06473666429519653f;
static const ai_float conv2d_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.003379718866199255f, 0.00501203490421176f, 0.002339448081329465f, 0.0026061194948852062f, 0.0026300710160285234f, 0.0023522712290287018f, 0.002518557710573077f, 0.0039031924679875374f, 0.002035001292824745f, 0.0018481320003047585f, 0.002209671540185809f, 0.003605149919167161f, 0.002473788568750024f, 0.0020140386186540127f, 0.0022474664729088545f, 0.0029628591146320105f, 0.002373701659962535f, 0.002814321545884013f, 0.002308897441253066f, 0.0022715970408171415f, 0.002465309575200081f, 0.002356409328058362f, 0.0016834903508424759f, 0.0019680154509842396f, 0.002554822014644742f, 0.002235539723187685f, 0.0029261154122650623f, 0.0034499792382121086f, 0.002769024344161153f, 0.003017824376001954f, 0.002619731705635786f, 0.002662966027855873f, 0.002988469321280718f, 0.002666214946657419f, 0.0022594567853957415f, 0.0019648184534162283f, 0.002945967484265566f, 0.0033312684390693903f, 0.0020211415830999613f, 0.003016366623342037f, 0.0023261404130607843f, 0.0018760828534141183f, 0.001803418854251504f, 0.0020383112132549286f, 0.0034728879109025f, 0.003374172141775489f, 0.002673728857189417f, 0.0024777052458375692f, 0.003469746559858322f, 0.0025198752991855145f, 0.0018183437641710043f, 0.0036727972328662872f, 0.0027431941125541925f, 0.0022187698632478714f, 0.002356862183660269f, 0.0024655181914567947f, 0.0024013123475015163f, 0.0016316049732267857f, 0.0021695434115827084f, 0.0020327193196862936f, 0.002745786914601922f, 0.0022390929516404867f, 0.0018260114593431354f, 0.002991237910464406f);
static const ai_layer_format_type conv2d_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 conv2d_0_t_out_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_0_t_out_0_shape_h_const_u16 = 97;

static const ai_i8 conv2d_1_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_1_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_1_pad_before_t_in_0_shape_h_const_u32 = 97;

static const ai_u16 conv2d_1_t_in_0_shape_w_const_u16 = 42;
static const ai_u16 conv2d_1_t_in_0_shape_h_const_u16 = 99;
static const ai_u16 conv2d_1_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_1_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_1_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_1_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_1_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_1_t_in_0_fmt_scale_const_f32 = 0.06473666429519653f;
static const ai_float conv2d_1_t_out_0_fmt_scale_const_f32 = 0.09375828504562378f;
static const ai_float conv2d_1_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.007955043576657772f, 0.011589824222028255f, 0.007430520374327898f, 0.005500090774148703f, 0.0061912317760288715f, 0.008052761666476727f, 0.008532091043889523f, 0.008396966382861137f, 0.004317965358495712f, 0.0043557691387832165f, 0.008480383083224297f, 0.004054273013025522f, 0.011727292090654373f, 0.006634620483964682f, 0.011085579171776772f, 0.008120597340166569f, 0.005288883578032255f, 0.007089500315487385f, 0.0066717700101435184f, 0.00798010267317295f, 0.007004601415246725f, 0.006252605002373457f, 0.0059328582137823105f, 0.006259569898247719f, 0.00524664344266057f, 0.004594600759446621f, 0.006820132490247488f, 0.004877476952970028f, 0.0060791741125285625f, 0.004892417695373297f, 0.0049101305194199085f, 0.007838993333280087f, 0.003748679766431451f, 0.009347667917609215f, 0.00654310267418623f, 0.004854499828070402f, 0.011196145787835121f, 0.006571403704583645f, 0.006804271601140499f, 0.00771437818184495f, 0.007897689938545227f, 0.006016245111823082f, 0.0038923213724046946f, 0.004851124715059996f, 0.004776857327669859f, 0.010055660270154476f, 0.01071916427463293f, 0.006113739684224129f, 0.007518543861806393f, 0.0037278877571225166f, 0.006955626420676708f, 0.010683028027415276f, 0.007555737625807524f, 0.005310337990522385f, 0.007777519058436155f, 0.007901286706328392f, 0.0063433474861085415f, 0.00792683195322752f, 0.004987231455743313f, 0.004625080619007349f, 0.011209617368876934f, 0.005024549085646868f, 0.0067068361677229404f, 0.009747442789375782f);
static const ai_u16 conv2d_1_t_out_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_1_t_out_0_shape_h_const_u16 = 97;






static const ai_u16 conv2d_11_t_in_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_11_t_in_0_shape_h_const_u16 = 97;
static const ai_u16 conv2d_11_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_11_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_11_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_11_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_11_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_11_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_11_t_in_0_fmt_scale_const_f32 = 0.055396609008312225f;
static const ai_float conv2d_11_t_out_0_fmt_scale_const_f32 = 0.07973260432481766f;
static const ai_float conv2d_11_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006840104702860117f, 0.005315348040312529f, 0.007317858282476664f, 0.006879593711346388f, 0.006198863033205271f, 0.006113993935286999f, 0.0069466992281377316f, 0.005552767310291529f, 0.008238544687628746f, 0.0065431720577180386f, 0.006007784977555275f, 0.0063207088969647884f, 0.007268550805747509f, 0.008837983943521976f, 0.006348450668156147f, 0.007273568771779537f, 0.005714839324355125f, 0.004887523129582405f, 0.00590280769392848f, 0.006463197059929371f, 0.006649165414273739f, 0.006751764565706253f, 0.007424592040479183f, 0.008041299879550934f, 0.005074706859886646f, 0.007104959338903427f, 0.00730068190023303f, 0.006376740522682667f, 0.007828730158507824f, 0.006410305853933096f, 0.006112847942858934f, 0.0062889703549444675f, 0.007976259104907513f, 0.007698826026171446f, 0.007179196458309889f, 0.007514036260545254f, 0.006274426821619272f, 0.00774943083524704f, 0.009178179316222668f, 0.006014952901750803f, 0.00600882014259696f, 0.008510812185704708f, 0.0064993989653885365f, 0.00720738572999835f, 0.007857607677578926f, 0.006748344283550978f, 0.006082573439925909f, 0.006389756221324205f, 0.008225003257393837f, 0.00694535207003355f, 0.006268206983804703f, 0.006588596850633621f, 0.006634164601564407f, 0.005810268223285675f, 0.004155758302658796f, 0.006057800725102425f, 0.005656519439071417f, 0.0074287462048232555f, 0.005982455797493458f, 0.00606697890907526f, 0.007018634583801031f, 0.006440973840653896f, 0.007561197504401207f, 0.00589927239343524f);
static const ai_layer_format_type conv2d_11_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_12_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_12_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_12_pad_before_t_in_0_shape_h_const_u32 = 97;

static const ai_u16 conv2d_12_t_in_0_shape_w_const_u16 = 42;
static const ai_u16 conv2d_12_t_in_0_shape_h_const_u16 = 99;
static const ai_u16 conv2d_12_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_12_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_12_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_12_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_12_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_12_t_in_0_fmt_scale_const_f32 = 0.07973260432481766f;
static const ai_float conv2d_12_t_out_0_fmt_scale_const_f32 = 0.12319736927747726f;
static const ai_float conv2d_12_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0037436597049236298f, 0.007467222865670919f, 0.004725821316242218f, 0.005196757614612579f, 0.00558102922514081f, 0.003318812232464552f, 0.006983148865401745f, 0.006400152109563351f, 0.007142684422433376f, 0.005655737593770027f, 0.006707936059683561f, 0.007403634954243898f, 0.008074321784079075f, 0.008123000152409077f, 0.008143153041601181f, 0.003499799408018589f, 0.007405026815831661f, 0.006080572959035635f, 0.007554050534963608f, 0.004115401301532984f, 0.00830839853733778f, 0.00489211268723011f, 0.005088065750896931f, 0.007768675684928894f, 0.010448761284351349f, 0.007549748755991459f, 0.006112845614552498f, 0.007974770851433277f, 0.004581764806061983f, 0.004285048693418503f, 0.004984445404261351f, 0.005518007092177868f, 0.005820968188345432f, 0.00860917754471302f, 0.0051251971162855625f, 0.005229262635111809f, 0.013188377022743225f, 0.006350039038807154f, 0.007027376443147659f, 0.0067803156562149525f, 0.007753613870590925f, 0.00573009205982089f, 0.0045600528828799725f, 0.003459265921264887f, 0.0064399586990475655f, 0.0064168889075517654f, 0.007843582890927792f, 0.004202453885227442f, 0.006436576601117849f, 0.006043223664164543f, 0.00835411250591278f, 0.005217652767896652f, 0.00675346190109849f, 0.005003433208912611f, 0.009336802177131176f, 0.004313922021538019f, 0.009214723482728004f, 0.007426926400512457f, 0.007578232791274786f, 0.004539633169770241f, 0.004754181485623121f, 0.007218658458441496f, 0.010142723098397255f, 0.006798468064516783f);
static const ai_u16 conv2d_12_t_out_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_12_t_out_0_shape_h_const_u16 = 97;






static const ai_u16 conv2d_22_t_in_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_22_t_in_0_shape_h_const_u16 = 97;
static const ai_u16 conv2d_22_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_22_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_22_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_22_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_22_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_22_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_22_t_in_0_fmt_scale_const_f32 = 0.05964299291372299f;
static const ai_float conv2d_22_t_out_0_fmt_scale_const_f32 = 0.08547744154930115f;
static const ai_float conv2d_22_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006004755850881338f, 0.008417675271630287f, 0.006850045640021563f, 0.005825494881719351f, 0.005203653126955032f, 0.0075075202621519566f, 0.008191382512450218f, 0.007089311257004738f, 0.005903063807636499f, 0.005874240305274725f, 0.005377373192459345f, 0.005378463305532932f, 0.005835597403347492f, 0.006568887736648321f, 0.005271690897643566f, 0.005637511145323515f, 0.0057841334491968155f, 0.005499471444636583f, 0.006405156571418047f, 0.0083463154733181f, 0.007552292197942734f, 0.006935963872820139f, 0.006825277116149664f, 0.00590705219656229f, 0.006646126974374056f, 0.00526616582646966f, 0.006380368955433369f, 0.006081945728510618f, 0.003868126543238759f, 0.004472269210964441f, 0.006475399248301983f, 0.004695660434663296f, 0.00499543733894825f, 0.007134603336453438f, 0.007587060332298279f, 0.005682966206222773f, 0.008436061441898346f, 0.007639343850314617f, 0.007431994192302227f, 0.007523197215050459f, 0.005504400469362736f, 0.007073301821947098f, 0.004613741301000118f, 0.0061930823139846325f, 0.004921425599604845f, 0.006273656152188778f, 0.006547261029481888f, 0.005376247689127922f, 0.00880027562379837f, 0.005998183041810989f, 0.00607368815690279f, 0.0060438066720962524f, 0.007406734861433506f, 0.00659313565120101f, 0.0060102795250713825f, 0.007502465043216944f, 0.007120916619896889f, 0.008070473559200764f, 0.004157551098614931f, 0.007982972078025341f, 0.007004645187407732f, 0.0064192647114396095f, 0.005478539504110813f, 0.006878511514514685f);
static const ai_layer_format_type conv2d_22_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_23_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_23_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_23_pad_before_t_in_0_shape_h_const_u32 = 97;

static const ai_u16 conv2d_23_t_in_0_shape_w_const_u16 = 42;
static const ai_u16 conv2d_23_t_in_0_shape_h_const_u16 = 99;
static const ai_u16 conv2d_23_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_23_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_23_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_23_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_23_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_23_t_in_0_fmt_scale_const_f32 = 0.08547744154930115f;
static const ai_float conv2d_23_t_out_0_fmt_scale_const_f32 = 0.12380179762840271f;
static const ai_float conv2d_23_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.005842840299010277f, 0.005724110174924135f, 0.004305131267756224f, 0.0042140306904911995f, 0.007700716145336628f, 0.006664731074124575f, 0.005754097364842892f, 0.0049505881033837795f, 0.004186662379652262f, 0.005673675332218409f, 0.0036939389538019896f, 0.009295249357819557f, 0.010484712198376656f, 0.005943410098552704f, 0.004647129215300083f, 0.007412029430270195f, 0.004331299569457769f, 0.004702358972281218f, 0.006547710858285427f, 0.003994275815784931f, 0.0058075059205293655f, 0.0037296158261597157f, 0.005394292995333672f, 0.0049391514621675014f, 0.004512122366577387f, 0.008937171660363674f, 0.010266905650496483f, 0.008405017666518688f, 0.008534830994904041f, 0.003432242665439844f, 0.009383215568959713f, 0.005788858979940414f, 0.0036843179259449244f, 0.003949316684156656f, 0.0049531469121575356f, 0.002899428131058812f, 0.005455829203128815f, 0.004405634943395853f, 0.004066689405590296f, 0.004358239937573671f, 0.003892711829394102f, 0.0065601542592048645f, 0.008103130385279655f, 0.005518591031432152f, 0.009753831662237644f, 0.004966093227267265f, 0.0077105192467570305f, 0.005911769345402718f, 0.005269917659461498f, 0.011395237408578396f, 0.005864198785275221f, 0.0074707879684865475f, 0.006206171121448278f, 0.007620848249644041f, 0.007682058494538069f, 0.0046735769137740135f, 0.004359789192676544f, 0.0053546857088804245f, 0.007188998628407717f, 0.004791695158928633f, 0.003734682686626911f, 0.004763951059430838f, 0.004956135526299477f, 0.005470947828143835f);
static const ai_u16 conv2d_23_t_out_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_23_t_out_0_shape_h_const_u16 = 97;






static const ai_u16 conv2d_33_t_in_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_33_t_in_0_shape_h_const_u16 = 97;
static const ai_u16 conv2d_33_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_33_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_33_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_33_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_33_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_33_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_33_t_in_0_fmt_scale_const_f32 = 0.08252226561307907f;
static const ai_float conv2d_33_t_out_0_fmt_scale_const_f32 = 0.08685844391584396f;
static const ai_float conv2d_33_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0045811887830495834f, 0.00505949929356575f, 0.007066019810736179f, 0.0041011120192706585f, 0.007951886393129826f, 0.004565151408314705f, 0.004666117485612631f, 0.005844179540872574f, 0.004417370539158583f, 0.008426493033766747f, 0.004240070469677448f, 0.006573962513357401f, 0.005846617743372917f, 0.00606773653998971f, 0.006625676527619362f, 0.005976625252515078f, 0.006276235915720463f, 0.00552033819258213f, 0.005721108056604862f, 0.004064181819558144f, 0.003812548238784075f, 0.004748257342725992f, 0.004681479185819626f, 0.0072373636066913605f, 0.004553826060146093f, 0.005908708553761244f, 0.00584635604172945f, 0.005177893675863743f, 0.005814451724290848f, 0.004819529131054878f, 0.007888136431574821f, 0.007788418792188168f, 0.005854281131178141f, 0.004781294614076614f, 0.005805377382785082f, 0.0065773529931902885f, 0.006335727870464325f, 0.0035425180103629827f, 0.0065965792164206505f, 0.005697958171367645f, 0.004219755530357361f, 0.008204810321331024f, 0.004120432306081057f, 0.005586388520896435f, 0.008828802034258842f, 0.005666475743055344f, 0.004640417639166117f, 0.006940983701497316f, 0.007267100736498833f, 0.006552908103913069f, 0.00551622174680233f, 0.004997026640921831f, 0.005648552905768156f, 0.0055931550450623035f, 0.00503108836710453f, 0.007336233276873827f, 0.004795769229531288f, 0.007226156536489725f, 0.00593959866091609f, 0.006295887753367424f, 0.006406689528375864f, 0.0055101714096963406f, 0.004652711562812328f, 0.007401647977530956f);
static const ai_layer_format_type conv2d_33_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_34_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_34_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_34_pad_before_t_in_0_shape_h_const_u32 = 97;

static const ai_u16 conv2d_34_t_in_0_shape_w_const_u16 = 42;
static const ai_u16 conv2d_34_t_in_0_shape_h_const_u16 = 99;
static const ai_u16 conv2d_34_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_34_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_34_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_34_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_34_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_34_t_in_0_fmt_scale_const_f32 = 0.08685844391584396f;
static const ai_float conv2d_34_t_out_0_fmt_scale_const_f32 = 0.12345076352357864f;
static const ai_float conv2d_34_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.007628024555742741f, 0.004832415841519833f, 0.0033109551295638084f, 0.0038057935889810324f, 0.0049604326486587524f, 0.007164685521274805f, 0.004582424648106098f, 0.0038510265294462442f, 0.009308408945798874f, 0.007655838504433632f, 0.006127104628831148f, 0.0037996298633515835f, 0.0036368179135024548f, 0.005753183737397194f, 0.006317837163805962f, 0.00562281534075737f, 0.0051814354956150055f, 0.0035837378818541765f, 0.004645301960408688f, 0.010198521427810192f, 0.009194218553602695f, 0.004926730878651142f, 0.002633627038449049f, 0.0066532655619084835f, 0.004570297431200743f, 0.004768021870404482f, 0.003987047355622053f, 0.007970145903527737f, 0.002950044581666589f, 0.011885099112987518f, 0.0043831174261868f, 0.00685481820255518f, 0.0033765246625989676f, 0.005498634185642004f, 0.002673105336725712f, 0.0053975628688931465f, 0.0059572989121079445f, 0.004331236705183983f, 0.002482073847204447f, 0.005009329412132502f, 0.004330191295593977f, 0.006504307966679335f, 0.004109013359993696f, 0.012137478217482567f, 0.005200737621635199f, 0.004496276844292879f, 0.0064781708642840385f, 0.00545929092913866f, 0.0034968857653439045f, 0.010978166945278645f, 0.004975185729563236f, 0.006416527088731527f, 0.004890322219580412f, 0.011861058883368969f, 0.0036651368718594313f, 0.005881740711629391f, 0.0040749735198915005f, 0.006143227219581604f, 0.002737564267590642f, 0.006760610733181238f, 0.006598510313779116f, 0.010244383476674557f, 0.011153742671012878f, 0.004932425916194916f);
static const ai_u16 conv2d_34_t_out_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_34_t_out_0_shape_h_const_u16 = 97;






static const ai_u16 conv2d_44_t_in_0_shape_w_const_u16 = 40;
static const ai_u16 conv2d_44_t_in_0_shape_h_const_u16 = 97;
static const ai_u16 conv2d_44_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_44_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_44_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_44_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_44_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_44_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_44_t_in_0_fmt_scale_const_f32 = 0.04626816138625145f;
static const ai_float conv2d_44_t_out_0_fmt_scale_const_f32 = 0.11792958527803421f;
static const ai_float conv2d_44_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.009295482188463211f, 0.010199375450611115f, 0.006291594821959734f, 0.010358123108744621f, 0.008553768508136272f, 0.008327241986989975f, 0.01061850693076849f, 0.008174118585884571f, 0.010723480954766273f, 0.009389269165694714f, 0.009603387676179409f, 0.00820891372859478f, 0.008367330767214298f, 0.006728398613631725f, 0.012000205926597118f, 0.01097810734063387f, 0.00945441797375679f, 0.008569610305130482f, 0.010918421670794487f, 0.0076426067389547825f, 0.012165422551333904f, 0.011506450362503529f, 0.009770043194293976f, 0.0134515929967165f, 0.010240515694022179f, 0.009511121548712254f, 0.010384563356637955f, 0.0060405186377465725f, 0.011819968931376934f, 0.010416289791464806f, 0.007422858849167824f, 0.012450643815100193f, 0.012542652897536755f, 0.0101638687774539f, 0.010847678408026695f, 0.009372563101351261f, 0.011259340681135654f, 0.008216488175094128f, 0.008615936152637005f, 0.011356671340763569f, 0.009782661683857441f, 0.008526968769729137f, 0.009395611472427845f, 0.010858424007892609f, 0.007884805090725422f, 0.009869251400232315f, 0.014969700947403908f, 0.00978051032871008f, 0.011438211426138878f, 0.012028179131448269f, 0.010503875091671944f, 0.011232065968215466f, 0.007821972481906414f, 0.007070145569741726f, 0.008621917106211185f, 0.01353196520358324f, 0.008525963872671127f, 0.008505631238222122f, 0.007064523175358772f, 0.008272387087345123f, 0.008779416792094707f, 0.012029727920889854f, 0.011362023651599884f, 0.013668207451701164f);
static const ai_layer_format_type conv2d_44_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u32 nl_47_t_in_0_shape_ch_prod_const_u32 = 2;
STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN conv2d_0 */
  {
      const ai_i8* conv2d_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_inputs[0] + 0);
    const ai_i8* conv2d_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 0);
    const ai_i32* conv2d_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2560);
    ai_i8* conv2d_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 91648);
    ai_i16* conv2d_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 85472);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(0, 1, {(stai_ptr) conv2d_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(conv2d_0_t_in_0_ptr_const_s8, conv2d_0_t_in_0_shape_w_const_u16, conv2d_0_t_in_0_shape_h_const_u16, conv2d_0_t_in_0_shape_ch_const_u16, conv2d_0_t_weight_0_ptr_const_s8, conv2d_0_t_out_0_shape_ch_const_u16, conv2d_0_t_weight_0_shape_w_const_u16, conv2d_0_t_weight_0_shape_h_const_u16, conv2d_0_l_stride_1_const_u16, conv2d_0_l_stride_0_const_u16, conv2d_0_l_pad_W_0_const_s32, conv2d_0_l_pad_H_0_const_s32, conv2d_0_t_weight_1_ptr_const_s32, conv2d_0_t_in_0_fmt_zero_const_s8, conv2d_0_t_out_0_fmt_zero_const_s8, conv2d_0_t_in_0_fmt_scale_const_f32, conv2d_0_t_out_0_fmt_scale_const_f32, conv2d_0_t_weight_0_fmt_scale_const_f32, conv2d_0_l_out_ch_format_const_layer_format_type, conv2d_0_t_out_0_ptr_s8, conv2d_0_t_out_0_shape_w_const_u16, conv2d_0_t_out_0_shape_h_const_u16, 1, 6176, conv2d_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(0, 1, {(stai_ptr) conv2d_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_0 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_1_pad_before */
  {
      const ai_ptr conv2d_1_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 91648);
    ai_ptr conv2d_1_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 73856);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(1, 1, {(stai_ptr) conv2d_1_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_1_pad_before_t_in_0_ptr_const_ptr, conv2d_1_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_1_pad_before_v_pad_constant_value_const_s8), conv2d_1_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_1_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(2560), (ai_i32)(2688), (ai_i32)(2688), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(1, 1, {(stai_ptr) conv2d_1_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_1_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_1 */
  {
      const ai_i8* conv2d_1_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 73856);
    const ai_i8* conv2d_1_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2816);
    const ai_i32* conv2d_1_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3392);
    ai_i8* conv2d_1_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 71296);
    ai_i16* conv2d_1_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 339968);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(1, 1, {(stai_ptr) conv2d_1_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_1_t_in_0_ptr_const_s8, conv2d_1_t_in_0_shape_w_const_u16, conv2d_1_t_in_0_shape_h_const_u16, conv2d_1_t_in_0_shape_ch_const_u16, conv2d_1_t_weight_0_ptr_const_s8, conv2d_1_l_stride_1_const_u16, conv2d_1_l_stride_0_const_u16, conv2d_1_t_weight_1_ptr_const_s32, conv2d_1_t_in_0_fmt_zero_const_s8, conv2d_1_t_out_0_fmt_zero_const_s8, conv2d_1_t_in_0_fmt_scale_const_f32, conv2d_1_t_out_0_fmt_scale_const_f32, conv2d_1_t_weight_0_fmt_scale_const_f32, conv2d_1_t_out_0_ptr_s8, conv2d_1_t_out_0_shape_w_const_u16, conv2d_1_t_out_0_shape_h_const_u16, 0, 2369, conv2d_1_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(1, 1, {(stai_ptr) conv2d_1_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_1 */
  /* LITE_KERNEL_SECTION BEGIN pool_2 */
  {
    
  forward_lite_ap_integer_INT8_pool_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pool_2 */
  /* LITE_KERNEL_SECTION BEGIN gemm_3 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_3 */
  /* LITE_KERNEL_SECTION BEGIN gemm_4 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_4 */
  /* LITE_KERNEL_SECTION BEGIN nl_5 */
  {
    
  forward_lite_nl_integer_nl_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_5 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_10 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_10(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_10 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_11 */
  {
      const ai_i8* conv2d_11_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 71296);
    const ai_i8* conv2d_11_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 6016);
    const ai_i32* conv2d_11_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 10112);
    ai_i8* conv2d_11_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 68736);
    ai_i16* conv2d_11_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 319616);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(11, 1, {(stai_ptr) conv2d_11_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_11_t_in_0_ptr_const_s8, conv2d_11_t_in_0_shape_w_const_u16, conv2d_11_t_in_0_shape_h_const_u16, conv2d_11_l_stride_1_const_u16, conv2d_11_l_stride_0_const_u16, conv2d_11_t_in_0_shape_ch_const_u16, conv2d_11_t_weight_0_ptr_const_s8, conv2d_11_t_out_0_shape_ch_const_u16, conv2d_11_t_weight_1_ptr_const_s32, conv2d_11_t_in_0_fmt_zero_const_s8, conv2d_11_t_out_0_fmt_zero_const_s8, conv2d_11_t_in_0_fmt_scale_const_f32, conv2d_11_t_out_0_fmt_scale_const_f32, conv2d_11_t_weight_0_fmt_scale_const_f32, conv2d_11_l_out_ch_format_const_layer_format_type, conv2d_11_t_out_0_ptr_s8, 1, 896, conv2d_11_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(11, 1, {(stai_ptr) conv2d_11_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_11 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_12_pad_before */
  {
      const ai_ptr conv2d_12_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 68736);
    ai_ptr conv2d_12_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 50944);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(12, 1, {(stai_ptr) conv2d_12_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_12_pad_before_t_in_0_ptr_const_ptr, conv2d_12_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_12_pad_before_v_pad_constant_value_const_s8), conv2d_12_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_12_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(2560), (ai_i32)(2688), (ai_i32)(2688), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(12, 1, {(stai_ptr) conv2d_12_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_12_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_12 */
  {
      const ai_i8* conv2d_12_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 50944);
    const ai_i8* conv2d_12_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 10368);
    const ai_i32* conv2d_12_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 10944);
    ai_i8* conv2d_12_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 48384);
    ai_i16* conv2d_12_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 317056);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(12, 1, {(stai_ptr) conv2d_12_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_12_t_in_0_ptr_const_s8, conv2d_12_t_in_0_shape_w_const_u16, conv2d_12_t_in_0_shape_h_const_u16, conv2d_12_t_in_0_shape_ch_const_u16, conv2d_12_t_weight_0_ptr_const_s8, conv2d_12_l_stride_1_const_u16, conv2d_12_l_stride_0_const_u16, conv2d_12_t_weight_1_ptr_const_s32, conv2d_12_t_in_0_fmt_zero_const_s8, conv2d_12_t_out_0_fmt_zero_const_s8, conv2d_12_t_in_0_fmt_scale_const_f32, conv2d_12_t_out_0_fmt_scale_const_f32, conv2d_12_t_weight_0_fmt_scale_const_f32, conv2d_12_t_out_0_ptr_s8, conv2d_12_t_out_0_shape_w_const_u16, conv2d_12_t_out_0_shape_h_const_u16, 0, 2369, conv2d_12_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(12, 1, {(stai_ptr) conv2d_12_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_12 */
  /* LITE_KERNEL_SECTION BEGIN pool_13 */
  {
    
  forward_lite_ap_integer_INT8_pool_13(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pool_13 */
  /* LITE_KERNEL_SECTION BEGIN gemm_14 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_14(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_14 */
  /* LITE_KERNEL_SECTION BEGIN gemm_15 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_15(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_15 */
  /* LITE_KERNEL_SECTION BEGIN nl_16 */
  {
    
  forward_lite_nl_integer_nl_16(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_16 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_21 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_21(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_21 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_22 */
  {
      const ai_i8* conv2d_22_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 48384);
    const ai_i8* conv2d_22_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 13568);
    const ai_i32* conv2d_22_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 17664);
    ai_i8* conv2d_22_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 45824);
    ai_i16* conv2d_22_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 296704);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(22, 1, {(stai_ptr) conv2d_22_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_22_t_in_0_ptr_const_s8, conv2d_22_t_in_0_shape_w_const_u16, conv2d_22_t_in_0_shape_h_const_u16, conv2d_22_l_stride_1_const_u16, conv2d_22_l_stride_0_const_u16, conv2d_22_t_in_0_shape_ch_const_u16, conv2d_22_t_weight_0_ptr_const_s8, conv2d_22_t_out_0_shape_ch_const_u16, conv2d_22_t_weight_1_ptr_const_s32, conv2d_22_t_in_0_fmt_zero_const_s8, conv2d_22_t_out_0_fmt_zero_const_s8, conv2d_22_t_in_0_fmt_scale_const_f32, conv2d_22_t_out_0_fmt_scale_const_f32, conv2d_22_t_weight_0_fmt_scale_const_f32, conv2d_22_l_out_ch_format_const_layer_format_type, conv2d_22_t_out_0_ptr_s8, 1, 896, conv2d_22_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(22, 1, {(stai_ptr) conv2d_22_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_22 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_23_pad_before */
  {
      const ai_ptr conv2d_23_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 45824);
    ai_ptr conv2d_23_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 28032);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(23, 1, {(stai_ptr) conv2d_23_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_23_pad_before_t_in_0_ptr_const_ptr, conv2d_23_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_23_pad_before_v_pad_constant_value_const_s8), conv2d_23_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_23_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(2560), (ai_i32)(2688), (ai_i32)(2688), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(23, 1, {(stai_ptr) conv2d_23_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_23_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_23 */
  {
      const ai_i8* conv2d_23_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 28032);
    const ai_i8* conv2d_23_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 17920);
    const ai_i32* conv2d_23_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 18496);
    ai_i8* conv2d_23_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 25472);
    ai_i16* conv2d_23_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 294144);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(23, 1, {(stai_ptr) conv2d_23_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_23_t_in_0_ptr_const_s8, conv2d_23_t_in_0_shape_w_const_u16, conv2d_23_t_in_0_shape_h_const_u16, conv2d_23_t_in_0_shape_ch_const_u16, conv2d_23_t_weight_0_ptr_const_s8, conv2d_23_l_stride_1_const_u16, conv2d_23_l_stride_0_const_u16, conv2d_23_t_weight_1_ptr_const_s32, conv2d_23_t_in_0_fmt_zero_const_s8, conv2d_23_t_out_0_fmt_zero_const_s8, conv2d_23_t_in_0_fmt_scale_const_f32, conv2d_23_t_out_0_fmt_scale_const_f32, conv2d_23_t_weight_0_fmt_scale_const_f32, conv2d_23_t_out_0_ptr_s8, conv2d_23_t_out_0_shape_w_const_u16, conv2d_23_t_out_0_shape_h_const_u16, 0, 2369, conv2d_23_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(23, 1, {(stai_ptr) conv2d_23_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_23 */
  /* LITE_KERNEL_SECTION BEGIN pool_24 */
  {
    
  forward_lite_ap_integer_INT8_pool_24(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pool_24 */
  /* LITE_KERNEL_SECTION BEGIN gemm_25 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_25(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_25 */
  /* LITE_KERNEL_SECTION BEGIN gemm_26 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_26(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_26 */
  /* LITE_KERNEL_SECTION BEGIN nl_27 */
  {
    
  forward_lite_nl_integer_nl_27(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_27 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_32 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_32(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_32 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_33 */
  {
      const ai_i8* conv2d_33_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 25472);
    const ai_i8* conv2d_33_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21120);
    const ai_i32* conv2d_33_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 25216);
    ai_i8* conv2d_33_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 22912);
    ai_i16* conv2d_33_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 273792);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(33, 1, {(stai_ptr) conv2d_33_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_33_t_in_0_ptr_const_s8, conv2d_33_t_in_0_shape_w_const_u16, conv2d_33_t_in_0_shape_h_const_u16, conv2d_33_l_stride_1_const_u16, conv2d_33_l_stride_0_const_u16, conv2d_33_t_in_0_shape_ch_const_u16, conv2d_33_t_weight_0_ptr_const_s8, conv2d_33_t_out_0_shape_ch_const_u16, conv2d_33_t_weight_1_ptr_const_s32, conv2d_33_t_in_0_fmt_zero_const_s8, conv2d_33_t_out_0_fmt_zero_const_s8, conv2d_33_t_in_0_fmt_scale_const_f32, conv2d_33_t_out_0_fmt_scale_const_f32, conv2d_33_t_weight_0_fmt_scale_const_f32, conv2d_33_l_out_ch_format_const_layer_format_type, conv2d_33_t_out_0_ptr_s8, 1, 896, conv2d_33_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(33, 1, {(stai_ptr) conv2d_33_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_33 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_34_pad_before */
  {
      const ai_ptr conv2d_34_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 22912);
    ai_ptr conv2d_34_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 5120);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 1, {(stai_ptr) conv2d_34_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_34_pad_before_t_in_0_ptr_const_ptr, conv2d_34_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_34_pad_before_v_pad_constant_value_const_s8), conv2d_34_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_34_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(2560), (ai_i32)(2688), (ai_i32)(2688), (ai_i32)(64), (ai_i32)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, {(stai_ptr) conv2d_34_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_34_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_34 */
  {
      const ai_i8* conv2d_34_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5120);
    const ai_i8* conv2d_34_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 25472);
    const ai_i32* conv2d_34_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 26048);
    ai_i8* conv2d_34_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 2560);
    ai_i16* conv2d_34_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 271232);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 1, {(stai_ptr) conv2d_34_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_34_t_in_0_ptr_const_s8, conv2d_34_t_in_0_shape_w_const_u16, conv2d_34_t_in_0_shape_h_const_u16, conv2d_34_t_in_0_shape_ch_const_u16, conv2d_34_t_weight_0_ptr_const_s8, conv2d_34_l_stride_1_const_u16, conv2d_34_l_stride_0_const_u16, conv2d_34_t_weight_1_ptr_const_s32, conv2d_34_t_in_0_fmt_zero_const_s8, conv2d_34_t_out_0_fmt_zero_const_s8, conv2d_34_t_in_0_fmt_scale_const_f32, conv2d_34_t_out_0_fmt_scale_const_f32, conv2d_34_t_weight_0_fmt_scale_const_f32, conv2d_34_t_out_0_ptr_s8, conv2d_34_t_out_0_shape_w_const_u16, conv2d_34_t_out_0_shape_h_const_u16, 0, 2369, conv2d_34_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, {(stai_ptr) conv2d_34_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_34 */
  /* LITE_KERNEL_SECTION BEGIN pool_35 */
  {
    
  forward_lite_ap_integer_INT8_pool_35(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pool_35 */
  /* LITE_KERNEL_SECTION BEGIN gemm_36 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_36(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_36 */
  /* LITE_KERNEL_SECTION BEGIN gemm_37 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_37(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_37 */
  /* LITE_KERNEL_SECTION BEGIN nl_38 */
  {
    
  forward_lite_nl_integer_nl_38(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_38 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_43 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_43(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_43 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_44 */
  {
      const ai_i8* conv2d_44_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2560);
    const ai_i8* conv2d_44_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 28672);
    const ai_i32* conv2d_44_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 32768);
    ai_i8* conv2d_44_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* conv2d_44_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 250880);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(44, 1, {(stai_ptr) conv2d_44_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_44_t_in_0_ptr_const_s8, conv2d_44_t_in_0_shape_w_const_u16, conv2d_44_t_in_0_shape_h_const_u16, conv2d_44_l_stride_1_const_u16, conv2d_44_l_stride_0_const_u16, conv2d_44_t_in_0_shape_ch_const_u16, conv2d_44_t_weight_0_ptr_const_s8, conv2d_44_t_out_0_shape_ch_const_u16, conv2d_44_t_weight_1_ptr_const_s32, conv2d_44_t_in_0_fmt_zero_const_s8, conv2d_44_t_out_0_fmt_zero_const_s8, conv2d_44_t_in_0_fmt_scale_const_f32, conv2d_44_t_out_0_fmt_scale_const_f32, conv2d_44_t_weight_0_fmt_scale_const_f32, conv2d_44_l_out_ch_format_const_layer_format_type, conv2d_44_t_out_0_ptr_s8, 1, 896, conv2d_44_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(44, 1, {(stai_ptr) conv2d_44_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_44 */
  /* LITE_KERNEL_SECTION BEGIN pool_45 */
  {
    
  forward_lite_ap_integer_INT8_pool_45(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pool_45 */
  /* LITE_KERNEL_SECTION BEGIN gemm_46 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_46(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_46 */
  /* LITE_KERNEL_SECTION BEGIN nl_47 */
  {
      ai_i8* nl_47_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_outputs[0] + 0);
    const ai_i8* nl_47_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 148);
    ai_i32* nl_47_t_scratch_0_ptr_s32 = (ai_i32*)(net_ctx->_activations[0] + 152);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(47, 1, {(stai_ptr) nl_47_t_in_0_ptr_const_s8});
    
  forward_lite_nl_softmax_is8os8(nl_47_t_out_0_ptr_s8, nl_47_t_in_0_ptr_const_s8, nl_47_t_in_0_shape_ch_prod_const_u32, 1, 2, 1384688000, 23, -248, nl_47_t_scratch_0_ptr_s32);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(47, 1, {(stai_ptr) nl_47_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END nl_47 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 81592;

  net_ctx->_outputs[0] = activations[0] + 0;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

