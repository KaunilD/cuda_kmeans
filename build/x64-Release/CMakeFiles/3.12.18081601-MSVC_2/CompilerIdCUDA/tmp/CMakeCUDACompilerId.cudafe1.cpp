#pragma section("__nv_managed_data__")
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
#line 1 "CMakeCUDACompilerId.cu"
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false

#line 1
#line 62 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2/bin/../include\\cuda_runtime.h"
#pragma warning(push)
#pragma warning(disable: 4820)
#line 708 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\sal.h"
#pragma region Input Buffer SAL 1 compatibility macros
#line 1472
#pragma endregion Input Buffer SAL 1 compatibility macros
#line 2361 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\sal.h"
extern "C" {
#line 2967 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\sal.h"
}
#line 22 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\concurrencysal.h"
extern "C" {
#line 354 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\concurrencysal.h"
}
#line 15 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vadefs.h"
#pragma pack ( push, 8 )
#line 18
extern "C" {
#line 28 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vadefs.h"
typedef unsigned __int64 uintptr_t; 
#line 39 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vadefs.h"
typedef char *va_list; 
#line 112 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vadefs.h"
void __cdecl __va_start(va_list *, ...); 
#line 124 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vadefs.h"
}
#line 128 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vadefs.h"
extern "C++" {
#line 130
template< class _Ty> 
#line 131
struct __vcrt_va_list_is_reference { 
#line 133
enum: bool { __the_value}; 
#line 134
}; 
#line 136
template< class _Ty> 
#line 137
struct __vcrt_va_list_is_reference< _Ty &>  { 
#line 139
enum: bool { __the_value = '\001'}; 
#line 140
}; 
#line 142
template< class _Ty> 
#line 143
struct __vcrt_va_list_is_reference< _Ty &&>  { 
#line 145
enum: bool { __the_value = '\001'}; 
#line 146
}; 
#line 148
template< class _Ty> 
#line 149
struct __vcrt_assert_va_start_is_not_reference { 
#line 151
static_assert((!__vcrt_va_list_is_reference< _Ty> ::__the_value), "va_start argument must not have reference type and must not be parenthesized");
#line 153
}; 
#line 154
}
#line 164 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vadefs.h"
#pragma pack ( pop )
#line 83 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 180 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime.h"
typedef unsigned __int64 size_t; 
#include "crt/host_runtime.h"
#line 181
typedef __int64 ptrdiff_t; 
#line 182
typedef __int64 intptr_t; 
#line 190 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime.h"
typedef bool __vcrt_bool; 
#line 233 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime.h"
extern "C++" {
#line 235
template< class _CountofType, size_t _SizeOfArray> char (*__countof_helper(__unaligned _CountofType (& _Array)[_SizeOfArray]))[_SizeOfArray]; 
#line 239
}
#line 277 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime.h"
void __cdecl __security_init_cookie(); 
#line 283
void __cdecl __security_check_cookie(uintptr_t _StackCookie); 
#line 284
__declspec(noreturn) void __cdecl __report_gsfailure(uintptr_t _StackCookie); 
#line 288 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime.h"
extern uintptr_t __security_cookie; 
#line 296 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime.h"
}__pragma( pack ( pop )) 
#line 12 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 136 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt.h"
extern "C++" {
#line 138
template< bool _Enable, class _Ty> struct _CrtEnableIf; 
#line 141
template< class _Ty> 
#line 142
struct _CrtEnableIf< true, _Ty>  { 
#line 144
typedef _Ty _Type; 
#line 145
}; 
#line 146
}
#line 150 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt.h"
typedef bool __crt_bool; 
#line 278 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt.h"
void __cdecl _invalid_parameter_noinfo(); 
#line 279
__declspec(noreturn) void __cdecl _invalid_parameter_noinfo_noreturn(); 
#line 281
__declspec(noreturn) void __cdecl 
#line 282
_invoke_watson(const __wchar_t * _Expression, const __wchar_t * _FunctionName, const __wchar_t * _FileName, unsigned _LineNo, uintptr_t _Reserved); 
#line 510 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt.h"
typedef int errno_t; 
#line 511
typedef unsigned short wint_t; 
#line 512
typedef unsigned short wctype_t; 
#line 513
typedef long __time32_t; 
#line 514
typedef __int64 __time64_t; 
#line 521
typedef 
#line 516
struct __crt_locale_data_public { 
#line 518
const unsigned short *_locale_pctype; 
#line 519
int _locale_mb_cur_max; 
#line 520
unsigned _locale_lc_codepage; 
#line 521
} __crt_locale_data_public; 
#line 527
typedef 
#line 523
struct __crt_locale_pointers { 
#line 525
struct __crt_locale_data *locinfo; 
#line 526
struct __crt_multibyte_data *mbcinfo; 
#line 527
} __crt_locale_pointers; 
#line 529
typedef __crt_locale_pointers *_locale_t; 
#line 535
typedef 
#line 531
struct _Mbstatet { 
#line 533
unsigned long _Wchar; 
#line 534
unsigned short _Byte, _State; 
#line 535
} _Mbstatet; 
#line 537
typedef _Mbstatet mbstate_t; 
#line 551 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt.h"
typedef __time64_t time_t; 
#line 561 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt.h"
typedef size_t rsize_t; 
#line 2010 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt.h"
}__pragma( pack ( pop )) 
#line 61 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_types.h"
#if 0
#line 61
enum cudaRoundMode { 
#line 63
cudaRoundNearest, 
#line 64
cudaRoundZero, 
#line 65
cudaRoundPosInf, 
#line 66
cudaRoundMinInf
#line 67
}; 
#endif
#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 93
struct char1 { 
#line 95
signed char x; 
#line 96
}; 
#endif
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 98
struct uchar1 { 
#line 100
unsigned char x; 
#line 101
}; 
#endif
#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 104
struct __declspec(align(2)) char2 { 
#line 106
signed char x, y; 
#line 107
}; 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 109
struct __declspec(align(2)) uchar2 { 
#line 111
unsigned char x, y; 
#line 112
}; 
#endif
#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 114
struct char3 { 
#line 116
signed char x, y, z; 
#line 117
}; 
#endif
#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 119
struct uchar3 { 
#line 121
unsigned char x, y, z; 
#line 122
}; 
#endif
#line 124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 124
struct __declspec(align(4)) char4 { 
#line 126
signed char x, y, z, w; 
#line 127
}; 
#endif
#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 129
struct __declspec(align(4)) uchar4 { 
#line 131
unsigned char x, y, z, w; 
#line 132
}; 
#endif
#line 134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 134
struct short1 { 
#line 136
short x; 
#line 137
}; 
#endif
#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 139
struct ushort1 { 
#line 141
unsigned short x; 
#line 142
}; 
#endif
#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 144
struct __declspec(align(4)) short2 { 
#line 146
short x, y; 
#line 147
}; 
#endif
#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 149
struct __declspec(align(4)) ushort2 { 
#line 151
unsigned short x, y; 
#line 152
}; 
#endif
#line 154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 154
struct short3 { 
#line 156
short x, y, z; 
#line 157
}; 
#endif
#line 159 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 159
struct ushort3 { 
#line 161
unsigned short x, y, z; 
#line 162
}; 
#endif
#line 164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 164
struct __declspec(align(8)) short4 { short x; short y; short z; short w; }; 
#endif
#line 165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 165
struct __declspec(align(8)) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
#line 167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 167
struct int1 { 
#line 169
int x; 
#line 170
}; 
#endif
#line 172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 172
struct uint1 { 
#line 174
unsigned x; 
#line 175
}; 
#endif
#line 177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 177
struct __declspec(align(8)) int2 { int x; int y; }; 
#endif
#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 178
struct __declspec(align(8)) uint2 { unsigned x; unsigned y; }; 
#endif
#line 180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 180
struct int3 { 
#line 182
int x, y, z; 
#line 183
}; 
#endif
#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 185
struct uint3 { 
#line 187
unsigned x, y, z; 
#line 188
}; 
#endif
#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 190
struct __declspec(align(16)) int4 { 
#line 192
int x, y, z, w; 
#line 193
}; 
#endif
#line 195 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 195
struct __declspec(align(16)) uint4 { 
#line 197
unsigned x, y, z, w; 
#line 198
}; 
#endif
#line 200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 200
struct long1 { 
#line 202
long x; 
#line 203
}; 
#endif
#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 205
struct ulong1 { 
#line 207
unsigned long x; 
#line 208
}; 
#endif
#line 211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 211
struct __declspec(align(8)) long2 { long x; long y; }; 
#endif
#line 212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 212
struct __declspec(align(8)) ulong2 { unsigned long x; unsigned long y; }; 
#endif
#line 227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 227
struct long3 { 
#line 229
long x, y, z; 
#line 230
}; 
#endif
#line 232 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 232
struct ulong3 { 
#line 234
unsigned long x, y, z; 
#line 235
}; 
#endif
#line 237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 237
struct __declspec(align(16)) long4 { 
#line 239
long x, y, z, w; 
#line 240
}; 
#endif
#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 242
struct __declspec(align(16)) ulong4 { 
#line 244
unsigned long x, y, z, w; 
#line 245
}; 
#endif
#line 247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 247
struct float1 { 
#line 249
float x; 
#line 250
}; 
#endif
#line 269 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 269
struct __declspec(align(8)) float2 { float x; float y; }; 
#endif
#line 274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 274
struct float3 { 
#line 276
float x, y, z; 
#line 277
}; 
#endif
#line 279 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 279
struct __declspec(align(16)) float4 { 
#line 281
float x, y, z, w; 
#line 282
}; 
#endif
#line 284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 284
struct longlong1 { 
#line 286
__int64 x; 
#line 287
}; 
#endif
#line 289 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 289
struct ulonglong1 { 
#line 291
unsigned __int64 x; 
#line 292
}; 
#endif
#line 294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 294
struct __declspec(align(16)) longlong2 { 
#line 296
__int64 x, y; 
#line 297
}; 
#endif
#line 299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 299
struct __declspec(align(16)) ulonglong2 { 
#line 301
unsigned __int64 x, y; 
#line 302
}; 
#endif
#line 304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 304
struct longlong3 { 
#line 306
__int64 x, y, z; 
#line 307
}; 
#endif
#line 309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 309
struct ulonglong3 { 
#line 311
unsigned __int64 x, y, z; 
#line 312
}; 
#endif
#line 314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 314
struct __declspec(align(16)) longlong4 { 
#line 316
__int64 x, y, z, w; 
#line 317
}; 
#endif
#line 319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 319
struct __declspec(align(16)) ulonglong4 { 
#line 321
unsigned __int64 x, y, z, w; 
#line 322
}; 
#endif
#line 324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 324
struct double1 { 
#line 326
double x; 
#line 327
}; 
#endif
#line 329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 329
struct __declspec(align(16)) double2 { 
#line 331
double x, y; 
#line 332
}; 
#endif
#line 334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 334
struct double3 { 
#line 336
double x, y, z; 
#line 337
}; 
#endif
#line 339 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 339
struct __declspec(align(16)) double4 { 
#line 341
double x, y, z, w; 
#line 342
}; 
#endif
#line 356 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef char1 
#line 356
char1; 
#endif
#line 357 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef uchar1 
#line 357
uchar1; 
#endif
#line 358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef char2 
#line 358
char2; 
#endif
#line 359 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef uchar2 
#line 359
uchar2; 
#endif
#line 360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef char3 
#line 360
char3; 
#endif
#line 361 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef uchar3 
#line 361
uchar3; 
#endif
#line 362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef char4 
#line 362
char4; 
#endif
#line 363 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef uchar4 
#line 363
uchar4; 
#endif
#line 364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef short1 
#line 364
short1; 
#endif
#line 365 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ushort1 
#line 365
ushort1; 
#endif
#line 366 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef short2 
#line 366
short2; 
#endif
#line 367 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ushort2 
#line 367
ushort2; 
#endif
#line 368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef short3 
#line 368
short3; 
#endif
#line 369 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ushort3 
#line 369
ushort3; 
#endif
#line 370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef short4 
#line 370
short4; 
#endif
#line 371 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ushort4 
#line 371
ushort4; 
#endif
#line 372 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef int1 
#line 372
int1; 
#endif
#line 373 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef uint1 
#line 373
uint1; 
#endif
#line 374 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef int2 
#line 374
int2; 
#endif
#line 375 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef uint2 
#line 375
uint2; 
#endif
#line 376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef int3 
#line 376
int3; 
#endif
#line 377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef uint3 
#line 377
uint3; 
#endif
#line 378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef int4 
#line 378
int4; 
#endif
#line 379 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef uint4 
#line 379
uint4; 
#endif
#line 380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef long1 
#line 380
long1; 
#endif
#line 381 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ulong1 
#line 381
ulong1; 
#endif
#line 382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef long2 
#line 382
long2; 
#endif
#line 383 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ulong2 
#line 383
ulong2; 
#endif
#line 384 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef long3 
#line 384
long3; 
#endif
#line 385 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ulong3 
#line 385
ulong3; 
#endif
#line 386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef long4 
#line 386
long4; 
#endif
#line 387 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ulong4 
#line 387
ulong4; 
#endif
#line 388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef float1 
#line 388
float1; 
#endif
#line 389 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef float2 
#line 389
float2; 
#endif
#line 390 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef float3 
#line 390
float3; 
#endif
#line 391 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef float4 
#line 391
float4; 
#endif
#line 392 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef longlong1 
#line 392
longlong1; 
#endif
#line 393 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ulonglong1 
#line 393
ulonglong1; 
#endif
#line 394 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef longlong2 
#line 394
longlong2; 
#endif
#line 395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ulonglong2 
#line 395
ulonglong2; 
#endif
#line 396 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef longlong3 
#line 396
longlong3; 
#endif
#line 397 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ulonglong3 
#line 397
ulonglong3; 
#endif
#line 398 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef longlong4 
#line 398
longlong4; 
#endif
#line 399 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef ulonglong4 
#line 399
ulonglong4; 
#endif
#line 400 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef double1 
#line 400
double1; 
#endif
#line 401 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef double2 
#line 401
double2; 
#endif
#line 402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef double3 
#line 402
double3; 
#endif
#line 403 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef double4 
#line 403
double4; 
#endif
#line 411 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
#line 411
struct dim3 { 
#line 413
unsigned x, y, z; 
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
}; 
#endif
#line 421 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_types.h"
#if 0
typedef dim3 
#line 421
dim3; 
#endif
#line 13 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\limits.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 88 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\limits.h"
}__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stddef.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
namespace std { 
#line 21
typedef decltype((nullptr)) nullptr_t; 
#line 22
}
#line 24
using std::nullptr_t;
#line 31 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stddef.h"
int *__cdecl _errno(); 
#line 34
errno_t __cdecl _set_errno(int _Value); 
#line 35
errno_t __cdecl _get_errno(int * _Value); 
#line 51 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stddef.h"
extern unsigned long __cdecl __threadid(); 
#line 53
extern uintptr_t __cdecl __threadhandle(); 
#line 57
}__pragma( pack ( pop )) 
#line 171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 171
enum cudaError { 
#line 178
cudaSuccess, 
#line 184
cudaErrorMissingConfiguration, 
#line 190
cudaErrorMemoryAllocation, 
#line 196
cudaErrorInitializationError, 
#line 206
cudaErrorLaunchFailure, 
#line 215
cudaErrorPriorLaunchFailure, 
#line 226
cudaErrorLaunchTimeout, 
#line 235
cudaErrorLaunchOutOfResources, 
#line 241
cudaErrorInvalidDeviceFunction, 
#line 250
cudaErrorInvalidConfiguration, 
#line 256
cudaErrorInvalidDevice, 
#line 262
cudaErrorInvalidValue, 
#line 268
cudaErrorInvalidPitchValue, 
#line 274
cudaErrorInvalidSymbol, 
#line 279
cudaErrorMapBufferObjectFailed, 
#line 284
cudaErrorUnmapBufferObjectFailed, 
#line 290
cudaErrorInvalidHostPointer, 
#line 296
cudaErrorInvalidDevicePointer, 
#line 302
cudaErrorInvalidTexture, 
#line 308
cudaErrorInvalidTextureBinding, 
#line 315
cudaErrorInvalidChannelDescriptor, 
#line 321
cudaErrorInvalidMemcpyDirection, 
#line 331
cudaErrorAddressOfConstant, 
#line 340
cudaErrorTextureFetchFailed, 
#line 349
cudaErrorTextureNotBound, 
#line 358
cudaErrorSynchronizationError, 
#line 364
cudaErrorInvalidFilterSetting, 
#line 370
cudaErrorInvalidNormSetting, 
#line 378
cudaErrorMixedDeviceExecution, 
#line 385
cudaErrorCudartUnloading, 
#line 390
cudaErrorUnknown, 
#line 398
cudaErrorNotYetImplemented, 
#line 407
cudaErrorMemoryValueTooLarge, 
#line 414
cudaErrorInvalidResourceHandle, 
#line 422
cudaErrorNotReady, 
#line 429
cudaErrorInsufficientDriver, 
#line 442
cudaErrorSetOnActiveProcess, 
#line 448
cudaErrorInvalidSurface, 
#line 454
cudaErrorNoDevice, 
#line 460
cudaErrorECCUncorrectable, 
#line 465
cudaErrorSharedObjectSymbolNotFound, 
#line 470
cudaErrorSharedObjectInitFailed, 
#line 476
cudaErrorUnsupportedLimit, 
#line 482
cudaErrorDuplicateVariableName, 
#line 488
cudaErrorDuplicateTextureName, 
#line 494
cudaErrorDuplicateSurfaceName, 
#line 504
cudaErrorDevicesUnavailable, 
#line 509
cudaErrorInvalidKernelImage, 
#line 517
cudaErrorNoKernelImageForDevice, 
#line 530
cudaErrorIncompatibleDriverContext, 
#line 537
cudaErrorPeerAccessAlreadyEnabled, 
#line 544
cudaErrorPeerAccessNotEnabled, 
#line 550
cudaErrorDeviceAlreadyInUse = 54, 
#line 557
cudaErrorProfilerDisabled, 
#line 565
cudaErrorProfilerNotInitialized, 
#line 572
cudaErrorProfilerAlreadyStarted, 
#line 579
cudaErrorProfilerAlreadyStopped, 
#line 587
cudaErrorAssert, 
#line 594
cudaErrorTooManyPeers, 
#line 600
cudaErrorHostMemoryAlreadyRegistered, 
#line 606
cudaErrorHostMemoryNotRegistered, 
#line 611
cudaErrorOperatingSystem, 
#line 617
cudaErrorPeerAccessUnsupported, 
#line 624
cudaErrorLaunchMaxDepthExceeded, 
#line 632
cudaErrorLaunchFileScopedTex, 
#line 640
cudaErrorLaunchFileScopedSurf, 
#line 655
cudaErrorSyncDepthExceeded, 
#line 667
cudaErrorLaunchPendingCountExceeded, 
#line 672
cudaErrorNotPermitted, 
#line 678
cudaErrorNotSupported, 
#line 687
cudaErrorHardwareStackError, 
#line 695
cudaErrorIllegalInstruction, 
#line 704
cudaErrorMisalignedAddress, 
#line 715
cudaErrorInvalidAddressSpace, 
#line 723
cudaErrorInvalidPc, 
#line 731
cudaErrorIllegalAddress, 
#line 737
cudaErrorInvalidPtx, 
#line 742
cudaErrorInvalidGraphicsContext, 
#line 748
cudaErrorNvlinkUncorrectable, 
#line 755
cudaErrorJitCompilerNotFound, 
#line 764
cudaErrorCooperativeLaunchTooLarge, 
#line 769
cudaErrorStartupFailure = 127, 
#line 777
cudaErrorApiFailureBase = 10000
#line 778
}; 
#endif
#line 783 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 783
enum cudaChannelFormatKind { 
#line 785
cudaChannelFormatKindSigned, 
#line 786
cudaChannelFormatKindUnsigned, 
#line 787
cudaChannelFormatKindFloat, 
#line 788
cudaChannelFormatKindNone
#line 789
}; 
#endif
#line 794 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 794
struct cudaChannelFormatDesc { 
#line 796
int x; 
#line 797
int y; 
#line 798
int z; 
#line 799
int w; 
#line 800
cudaChannelFormatKind f; 
#line 801
}; 
#endif
#line 806 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
typedef struct cudaArray *cudaArray_t; 
#line 811
typedef const cudaArray *cudaArray_const_t; 
#line 813
struct cudaArray; 
#line 818
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
#line 823
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
#line 825
struct cudaMipmappedArray; 
#line 830
#if 0
#line 830
enum cudaMemoryType { 
#line 832
cudaMemoryTypeHost = 1, 
#line 833
cudaMemoryTypeDevice
#line 834
}; 
#endif
#line 839 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 839
enum cudaMemcpyKind { 
#line 841
cudaMemcpyHostToHost, 
#line 842
cudaMemcpyHostToDevice, 
#line 843
cudaMemcpyDeviceToHost, 
#line 844
cudaMemcpyDeviceToDevice, 
#line 845
cudaMemcpyDefault
#line 846
}; 
#endif
#line 853 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 853
struct cudaPitchedPtr { 
#line 855
void *ptr; 
#line 856
size_t pitch; 
#line 857
size_t xsize; 
#line 858
size_t ysize; 
#line 859
}; 
#endif
#line 866 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 866
struct cudaExtent { 
#line 868
size_t width; 
#line 869
size_t height; 
#line 870
size_t depth; 
#line 871
}; 
#endif
#line 878 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 878
struct cudaPos { 
#line 880
size_t x; 
#line 881
size_t y; 
#line 882
size_t z; 
#line 883
}; 
#endif
#line 888 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 888
struct cudaMemcpy3DParms { 
#line 890
cudaArray_t srcArray; 
#line 891
cudaPos srcPos; 
#line 892
cudaPitchedPtr srcPtr; 
#line 894
cudaArray_t dstArray; 
#line 895
cudaPos dstPos; 
#line 896
cudaPitchedPtr dstPtr; 
#line 898
cudaExtent extent; 
#line 899
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 900
}; 
#endif
#line 905 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 905
struct cudaMemcpy3DPeerParms { 
#line 907
cudaArray_t srcArray; 
#line 908
cudaPos srcPos; 
#line 909
cudaPitchedPtr srcPtr; 
#line 910
int srcDevice; 
#line 912
cudaArray_t dstArray; 
#line 913
cudaPos dstPos; 
#line 914
cudaPitchedPtr dstPtr; 
#line 915
int dstDevice; 
#line 917
cudaExtent extent; 
#line 918
}; 
#endif
#line 923 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
struct cudaGraphicsResource; 
#line 928
#if 0
#line 928
enum cudaGraphicsRegisterFlags { 
#line 930
cudaGraphicsRegisterFlagsNone, 
#line 931
cudaGraphicsRegisterFlagsReadOnly, 
#line 932
cudaGraphicsRegisterFlagsWriteDiscard, 
#line 933
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
#line 934
cudaGraphicsRegisterFlagsTextureGather = 8
#line 935
}; 
#endif
#line 940 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 940
enum cudaGraphicsMapFlags { 
#line 942
cudaGraphicsMapFlagsNone, 
#line 943
cudaGraphicsMapFlagsReadOnly, 
#line 944
cudaGraphicsMapFlagsWriteDiscard
#line 945
}; 
#endif
#line 950 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 950
enum cudaGraphicsCubeFace { 
#line 952
cudaGraphicsCubeFacePositiveX, 
#line 953
cudaGraphicsCubeFaceNegativeX, 
#line 954
cudaGraphicsCubeFacePositiveY, 
#line 955
cudaGraphicsCubeFaceNegativeY, 
#line 956
cudaGraphicsCubeFacePositiveZ, 
#line 957
cudaGraphicsCubeFaceNegativeZ
#line 958
}; 
#endif
#line 963 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 963
enum cudaResourceType { 
#line 965
cudaResourceTypeArray, 
#line 966
cudaResourceTypeMipmappedArray, 
#line 967
cudaResourceTypeLinear, 
#line 968
cudaResourceTypePitch2D
#line 969
}; 
#endif
#line 974 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 974
enum cudaResourceViewFormat { 
#line 976
cudaResViewFormatNone, 
#line 977
cudaResViewFormatUnsignedChar1, 
#line 978
cudaResViewFormatUnsignedChar2, 
#line 979
cudaResViewFormatUnsignedChar4, 
#line 980
cudaResViewFormatSignedChar1, 
#line 981
cudaResViewFormatSignedChar2, 
#line 982
cudaResViewFormatSignedChar4, 
#line 983
cudaResViewFormatUnsignedShort1, 
#line 984
cudaResViewFormatUnsignedShort2, 
#line 985
cudaResViewFormatUnsignedShort4, 
#line 986
cudaResViewFormatSignedShort1, 
#line 987
cudaResViewFormatSignedShort2, 
#line 988
cudaResViewFormatSignedShort4, 
#line 989
cudaResViewFormatUnsignedInt1, 
#line 990
cudaResViewFormatUnsignedInt2, 
#line 991
cudaResViewFormatUnsignedInt4, 
#line 992
cudaResViewFormatSignedInt1, 
#line 993
cudaResViewFormatSignedInt2, 
#line 994
cudaResViewFormatSignedInt4, 
#line 995
cudaResViewFormatHalf1, 
#line 996
cudaResViewFormatHalf2, 
#line 997
cudaResViewFormatHalf4, 
#line 998
cudaResViewFormatFloat1, 
#line 999
cudaResViewFormatFloat2, 
#line 1000
cudaResViewFormatFloat4, 
#line 1001
cudaResViewFormatUnsignedBlockCompressed1, 
#line 1002
cudaResViewFormatUnsignedBlockCompressed2, 
#line 1003
cudaResViewFormatUnsignedBlockCompressed3, 
#line 1004
cudaResViewFormatUnsignedBlockCompressed4, 
#line 1005
cudaResViewFormatSignedBlockCompressed4, 
#line 1006
cudaResViewFormatUnsignedBlockCompressed5, 
#line 1007
cudaResViewFormatSignedBlockCompressed5, 
#line 1008
cudaResViewFormatUnsignedBlockCompressed6H, 
#line 1009
cudaResViewFormatSignedBlockCompressed6H, 
#line 1010
cudaResViewFormatUnsignedBlockCompressed7
#line 1011
}; 
#endif
#line 1016 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1016
struct cudaResourceDesc { 
#line 1017
cudaResourceType resType; 
#line 1019
union { 
#line 1020
struct { 
#line 1021
cudaArray_t array; 
#line 1022
} array; 
#line 1023
struct { 
#line 1024
cudaMipmappedArray_t mipmap; 
#line 1025
} mipmap; 
#line 1026
struct { 
#line 1027
void *devPtr; 
#line 1028
cudaChannelFormatDesc desc; 
#line 1029
size_t sizeInBytes; 
#line 1030
} linear; 
#line 1031
struct { 
#line 1032
void *devPtr; 
#line 1033
cudaChannelFormatDesc desc; 
#line 1034
size_t width; 
#line 1035
size_t height; 
#line 1036
size_t pitchInBytes; 
#line 1037
} pitch2D; 
#line 1038
} res; 
#line 1039
}; 
#endif
#line 1044 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1044
struct cudaResourceViewDesc { 
#line 1046
cudaResourceViewFormat format; 
#line 1047
size_t width; 
#line 1048
size_t height; 
#line 1049
size_t depth; 
#line 1050
unsigned firstMipmapLevel; 
#line 1051
unsigned lastMipmapLevel; 
#line 1052
unsigned firstLayer; 
#line 1053
unsigned lastLayer; 
#line 1054
}; 
#endif
#line 1059 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1059
struct cudaPointerAttributes { 
#line 1065
cudaMemoryType memoryType; 
#line 1076
int device; 
#line 1082
void *devicePointer; 
#line 1088
void *hostPointer; 
#line 1093
int isManaged; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1094
}; 
#endif
#line 1099 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1099
struct cudaFuncAttributes { 
#line 1106
size_t sharedSizeBytes; 
#line 1112
size_t constSizeBytes; 
#line 1117
size_t localSizeBytes; 
#line 1124
int maxThreadsPerBlock; 
#line 1129
int numRegs; 
#line 1136
int ptxVersion; 
#line 1143
int binaryVersion; 
#line 1149
int cacheModeCA; 
#line 1156
int maxDynamicSharedSizeBytes; 
#line 1165
int preferredShmemCarveout; 
#line 1166
}; 
#endif
#line 1171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1171
enum cudaFuncAttribute { 
#line 1173
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
#line 1174
cudaFuncAttributePreferredSharedMemoryCarveout, 
#line 1175
cudaFuncAttributeMax
#line 1176
}; 
#endif
#line 1181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1181
enum cudaFuncCache { 
#line 1183
cudaFuncCachePreferNone, 
#line 1184
cudaFuncCachePreferShared, 
#line 1185
cudaFuncCachePreferL1, 
#line 1186
cudaFuncCachePreferEqual
#line 1187
}; 
#endif
#line 1193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1193
enum cudaSharedMemConfig { 
#line 1195
cudaSharedMemBankSizeDefault, 
#line 1196
cudaSharedMemBankSizeFourByte, 
#line 1197
cudaSharedMemBankSizeEightByte
#line 1198
}; 
#endif
#line 1203 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1203
enum cudaSharedCarveout { 
#line 1204
cudaSharedmemCarveoutDefault = (-1), 
#line 1205
cudaSharedmemCarveoutMaxShared = 100, 
#line 1206
cudaSharedmemCarveoutMaxL1 = 0
#line 1207
}; 
#endif
#line 1212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1212
enum cudaComputeMode { 
#line 1214
cudaComputeModeDefault, 
#line 1215
cudaComputeModeExclusive, 
#line 1216
cudaComputeModeProhibited, 
#line 1217
cudaComputeModeExclusiveProcess
#line 1218
}; 
#endif
#line 1223 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1223
enum cudaLimit { 
#line 1225
cudaLimitStackSize, 
#line 1226
cudaLimitPrintfFifoSize, 
#line 1227
cudaLimitMallocHeapSize, 
#line 1228
cudaLimitDevRuntimeSyncDepth, 
#line 1229
cudaLimitDevRuntimePendingLaunchCount
#line 1230
}; 
#endif
#line 1235 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1235
enum cudaMemoryAdvise { 
#line 1237
cudaMemAdviseSetReadMostly = 1, 
#line 1238
cudaMemAdviseUnsetReadMostly, 
#line 1239
cudaMemAdviseSetPreferredLocation, 
#line 1240
cudaMemAdviseUnsetPreferredLocation, 
#line 1241
cudaMemAdviseSetAccessedBy, 
#line 1242
cudaMemAdviseUnsetAccessedBy
#line 1243
}; 
#endif
#line 1248 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1248
enum cudaMemRangeAttribute { 
#line 1250
cudaMemRangeAttributeReadMostly = 1, 
#line 1251
cudaMemRangeAttributePreferredLocation, 
#line 1252
cudaMemRangeAttributeAccessedBy, 
#line 1253
cudaMemRangeAttributeLastPrefetchLocation
#line 1254
}; 
#endif
#line 1259 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1259
enum cudaOutputMode { 
#line 1261
cudaKeyValuePair, 
#line 1262
cudaCSV
#line 1263
}; 
#endif
#line 1268 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1268
enum cudaDeviceAttr { 
#line 1270
cudaDevAttrMaxThreadsPerBlock = 1, 
#line 1271
cudaDevAttrMaxBlockDimX, 
#line 1272
cudaDevAttrMaxBlockDimY, 
#line 1273
cudaDevAttrMaxBlockDimZ, 
#line 1274
cudaDevAttrMaxGridDimX, 
#line 1275
cudaDevAttrMaxGridDimY, 
#line 1276
cudaDevAttrMaxGridDimZ, 
#line 1277
cudaDevAttrMaxSharedMemoryPerBlock, 
#line 1278
cudaDevAttrTotalConstantMemory, 
#line 1279
cudaDevAttrWarpSize, 
#line 1280
cudaDevAttrMaxPitch, 
#line 1281
cudaDevAttrMaxRegistersPerBlock, 
#line 1282
cudaDevAttrClockRate, 
#line 1283
cudaDevAttrTextureAlignment, 
#line 1284
cudaDevAttrGpuOverlap, 
#line 1285
cudaDevAttrMultiProcessorCount, 
#line 1286
cudaDevAttrKernelExecTimeout, 
#line 1287
cudaDevAttrIntegrated, 
#line 1288
cudaDevAttrCanMapHostMemory, 
#line 1289
cudaDevAttrComputeMode, 
#line 1290
cudaDevAttrMaxTexture1DWidth, 
#line 1291
cudaDevAttrMaxTexture2DWidth, 
#line 1292
cudaDevAttrMaxTexture2DHeight, 
#line 1293
cudaDevAttrMaxTexture3DWidth, 
#line 1294
cudaDevAttrMaxTexture3DHeight, 
#line 1295
cudaDevAttrMaxTexture3DDepth, 
#line 1296
cudaDevAttrMaxTexture2DLayeredWidth, 
#line 1297
cudaDevAttrMaxTexture2DLayeredHeight, 
#line 1298
cudaDevAttrMaxTexture2DLayeredLayers, 
#line 1299
cudaDevAttrSurfaceAlignment, 
#line 1300
cudaDevAttrConcurrentKernels, 
#line 1301
cudaDevAttrEccEnabled, 
#line 1302
cudaDevAttrPciBusId, 
#line 1303
cudaDevAttrPciDeviceId, 
#line 1304
cudaDevAttrTccDriver, 
#line 1305
cudaDevAttrMemoryClockRate, 
#line 1306
cudaDevAttrGlobalMemoryBusWidth, 
#line 1307
cudaDevAttrL2CacheSize, 
#line 1308
cudaDevAttrMaxThreadsPerMultiProcessor, 
#line 1309
cudaDevAttrAsyncEngineCount, 
#line 1310
cudaDevAttrUnifiedAddressing, 
#line 1311
cudaDevAttrMaxTexture1DLayeredWidth, 
#line 1312
cudaDevAttrMaxTexture1DLayeredLayers, 
#line 1313
cudaDevAttrMaxTexture2DGatherWidth = 45, 
#line 1314
cudaDevAttrMaxTexture2DGatherHeight, 
#line 1315
cudaDevAttrMaxTexture3DWidthAlt, 
#line 1316
cudaDevAttrMaxTexture3DHeightAlt, 
#line 1317
cudaDevAttrMaxTexture3DDepthAlt, 
#line 1318
cudaDevAttrPciDomainId, 
#line 1319
cudaDevAttrTexturePitchAlignment, 
#line 1320
cudaDevAttrMaxTextureCubemapWidth, 
#line 1321
cudaDevAttrMaxTextureCubemapLayeredWidth, 
#line 1322
cudaDevAttrMaxTextureCubemapLayeredLayers, 
#line 1323
cudaDevAttrMaxSurface1DWidth, 
#line 1324
cudaDevAttrMaxSurface2DWidth, 
#line 1325
cudaDevAttrMaxSurface2DHeight, 
#line 1326
cudaDevAttrMaxSurface3DWidth, 
#line 1327
cudaDevAttrMaxSurface3DHeight, 
#line 1328
cudaDevAttrMaxSurface3DDepth, 
#line 1329
cudaDevAttrMaxSurface1DLayeredWidth, 
#line 1330
cudaDevAttrMaxSurface1DLayeredLayers, 
#line 1331
cudaDevAttrMaxSurface2DLayeredWidth, 
#line 1332
cudaDevAttrMaxSurface2DLayeredHeight, 
#line 1333
cudaDevAttrMaxSurface2DLayeredLayers, 
#line 1334
cudaDevAttrMaxSurfaceCubemapWidth, 
#line 1335
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
#line 1336
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
#line 1337
cudaDevAttrMaxTexture1DLinearWidth, 
#line 1338
cudaDevAttrMaxTexture2DLinearWidth, 
#line 1339
cudaDevAttrMaxTexture2DLinearHeight, 
#line 1340
cudaDevAttrMaxTexture2DLinearPitch, 
#line 1341
cudaDevAttrMaxTexture2DMipmappedWidth, 
#line 1342
cudaDevAttrMaxTexture2DMipmappedHeight, 
#line 1343
cudaDevAttrComputeCapabilityMajor, 
#line 1344
cudaDevAttrComputeCapabilityMinor, 
#line 1345
cudaDevAttrMaxTexture1DMipmappedWidth, 
#line 1346
cudaDevAttrStreamPrioritiesSupported, 
#line 1347
cudaDevAttrGlobalL1CacheSupported, 
#line 1348
cudaDevAttrLocalL1CacheSupported, 
#line 1349
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
#line 1350
cudaDevAttrMaxRegistersPerMultiprocessor, 
#line 1351
cudaDevAttrManagedMemory, 
#line 1352
cudaDevAttrIsMultiGpuBoard, 
#line 1353
cudaDevAttrMultiGpuBoardGroupID, 
#line 1354
cudaDevAttrHostNativeAtomicSupported, 
#line 1355
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
#line 1356
cudaDevAttrPageableMemoryAccess, 
#line 1357
cudaDevAttrConcurrentManagedAccess, 
#line 1358
cudaDevAttrComputePreemptionSupported, 
#line 1359
cudaDevAttrCanUseHostPointerForRegisteredMem, 
#line 1360
cudaDevAttrReserved92, 
#line 1361
cudaDevAttrReserved93, 
#line 1362
cudaDevAttrReserved94, 
#line 1363
cudaDevAttrCooperativeLaunch, 
#line 1364
cudaDevAttrCooperativeMultiDeviceLaunch, 
#line 1365
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
#line 1366
cudaDevAttrCanFlushRemoteWrites, 
#line 1367
cudaDevAttrHostRegisterSupported, 
#line 1368
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
#line 1369
cudaDevAttrDirectManagedMemAccessFromHost
#line 1370
}; 
#endif
#line 1376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1376
enum cudaDeviceP2PAttr { 
#line 1377
cudaDevP2PAttrPerformanceRank = 1, 
#line 1378
cudaDevP2PAttrAccessSupported, 
#line 1379
cudaDevP2PAttrNativeAtomicSupported, 
#line 1380
cudaDevP2PAttrCudaArrayAccessSupported
#line 1381
}; 
#endif
#line 1385 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1385
struct cudaDeviceProp { 
#line 1387
char name[256]; 
#line 1388
size_t totalGlobalMem; 
#line 1389
size_t sharedMemPerBlock; 
#line 1390
int regsPerBlock; 
#line 1391
int warpSize; 
#line 1392
size_t memPitch; 
#line 1393
int maxThreadsPerBlock; 
#line 1394
int maxThreadsDim[3]; 
#line 1395
int maxGridSize[3]; 
#line 1396
int clockRate; 
#line 1397
size_t totalConstMem; 
#line 1398
int major; 
#line 1399
int minor; 
#line 1400
size_t textureAlignment; 
#line 1401
size_t texturePitchAlignment; 
#line 1402
int deviceOverlap; 
#line 1403
int multiProcessorCount; 
#line 1404
int kernelExecTimeoutEnabled; 
#line 1405
int integrated; 
#line 1406
int canMapHostMemory; 
#line 1407
int computeMode; 
#line 1408
int maxTexture1D; 
#line 1409
int maxTexture1DMipmap; 
#line 1410
int maxTexture1DLinear; 
#line 1411
int maxTexture2D[2]; 
#line 1412
int maxTexture2DMipmap[2]; 
#line 1413
int maxTexture2DLinear[3]; 
#line 1414
int maxTexture2DGather[2]; 
#line 1415
int maxTexture3D[3]; 
#line 1416
int maxTexture3DAlt[3]; 
#line 1417
int maxTextureCubemap; 
#line 1418
int maxTexture1DLayered[2]; 
#line 1419
int maxTexture2DLayered[3]; 
#line 1420
int maxTextureCubemapLayered[2]; 
#line 1421
int maxSurface1D; 
#line 1422
int maxSurface2D[2]; 
#line 1423
int maxSurface3D[3]; 
#line 1424
int maxSurface1DLayered[2]; 
#line 1425
int maxSurface2DLayered[3]; 
#line 1426
int maxSurfaceCubemap; 
#line 1427
int maxSurfaceCubemapLayered[2]; 
#line 1428
size_t surfaceAlignment; 
#line 1429
int concurrentKernels; 
#line 1430
int ECCEnabled; 
#line 1431
int pciBusID; 
#line 1432
int pciDeviceID; 
#line 1433
int pciDomainID; 
#line 1434
int tccDriver; 
#line 1435
int asyncEngineCount; 
#line 1436
int unifiedAddressing; 
#line 1437
int memoryClockRate; 
#line 1438
int memoryBusWidth; 
#line 1439
int l2CacheSize; 
#line 1440
int maxThreadsPerMultiProcessor; 
#line 1441
int streamPrioritiesSupported; 
#line 1442
int globalL1CacheSupported; 
#line 1443
int localL1CacheSupported; 
#line 1444
size_t sharedMemPerMultiprocessor; 
#line 1445
int regsPerMultiprocessor; 
#line 1446
int managedMemory; 
#line 1447
int isMultiGpuBoard; 
#line 1448
int multiGpuBoardGroupID; 
#line 1449
int hostNativeAtomicSupported; 
#line 1450
int singleToDoublePrecisionPerfRatio; 
#line 1451
int pageableMemoryAccess; 
#line 1452
int concurrentManagedAccess; 
#line 1453
int computePreemptionSupported; 
#line 1454
int canUseHostPointerForRegisteredMem; 
#line 1455
int cooperativeLaunch; 
#line 1456
int cooperativeMultiDeviceLaunch; 
#line 1457
size_t sharedMemPerBlockOptin; 
#line 1458
int pageableMemoryAccessUsesHostPageTables; 
#line 1459
int directManagedMemAccessFromHost; 
#line 1460
}; 
#endif
#line 1550 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
typedef 
#line 1547
struct cudaIpcEventHandle_st { 
#line 1549
char reserved[64]; 
#line 1550
} cudaIpcEventHandle_t; 
#endif
#line 1558 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
typedef 
#line 1555
struct cudaIpcMemHandle_st { 
#line 1557
char reserved[64]; 
#line 1558
} cudaIpcMemHandle_t; 
#endif
#line 1569 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
typedef cudaError 
#line 1569
cudaError_t; 
#endif
#line 1574 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
typedef struct CUstream_st *
#line 1574
cudaStream_t; 
#endif
#line 1579 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
typedef struct CUevent_st *
#line 1579
cudaEvent_t; 
#endif
#line 1584 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
typedef cudaGraphicsResource *
#line 1584
cudaGraphicsResource_t; 
#endif
#line 1589 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
typedef struct CUuuid_st 
#line 1589
cudaUUID_t; 
#endif
#line 1594 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
typedef cudaOutputMode 
#line 1594
cudaOutputMode_t; 
#endif
#line 1599 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1599
enum cudaCGScope { 
#line 1600
cudaCGScopeInvalid, 
#line 1601
cudaCGScopeGrid, 
#line 1602
cudaCGScopeMultiGrid
#line 1603
}; 
#endif
#line 1608 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_types.h"
#if 0
#line 1608
struct cudaLaunchParams { 
#line 1610
void *func; 
#line 1611
dim3 gridDim; 
#line 1612
dim3 blockDim; 
#line 1613
void **args; 
#line 1614
size_t sharedMem; 
#line 1615
cudaStream_t stream; 
#line 1616
}; 
#endif
#line 84 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_types.h"
#if 0
#line 84
enum cudaSurfaceBoundaryMode { 
#line 86
cudaBoundaryModeZero, 
#line 87
cudaBoundaryModeClamp, 
#line 88
cudaBoundaryModeTrap
#line 89
}; 
#endif
#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_types.h"
#if 0
#line 94
enum cudaSurfaceFormatMode { 
#line 96
cudaFormatModeForced, 
#line 97
cudaFormatModeAuto
#line 98
}; 
#endif
#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_types.h"
#if 0
#line 103
struct surfaceReference { 
#line 108
cudaChannelFormatDesc channelDesc; 
#line 109
}; 
#endif
#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_types.h"
#if 0
typedef unsigned __int64 
#line 114
cudaSurfaceObject_t; 
#endif
#line 84 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_types.h"
#if 0
#line 84
enum cudaTextureAddressMode { 
#line 86
cudaAddressModeWrap, 
#line 87
cudaAddressModeClamp, 
#line 88
cudaAddressModeMirror, 
#line 89
cudaAddressModeBorder
#line 90
}; 
#endif
#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_types.h"
#if 0
#line 95
enum cudaTextureFilterMode { 
#line 97
cudaFilterModePoint, 
#line 98
cudaFilterModeLinear
#line 99
}; 
#endif
#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_types.h"
#if 0
#line 104
enum cudaTextureReadMode { 
#line 106
cudaReadModeElementType, 
#line 107
cudaReadModeNormalizedFloat
#line 108
}; 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_types.h"
#if 0
#line 113
struct textureReference { 
#line 118
int normalized; 
#line 122
cudaTextureFilterMode filterMode; 
#line 126
cudaTextureAddressMode addressMode[3]; 
#line 130
cudaChannelFormatDesc channelDesc; 
#line 134
int sRGB; 
#line 138
unsigned maxAnisotropy; 
#line 142
cudaTextureFilterMode mipmapFilterMode; 
#line 146
float mipmapLevelBias; 
#line 150
float minMipmapLevelClamp; 
#line 154
float maxMipmapLevelClamp; 
#line 155
int __cudaReserved[15]; 
#line 156
}; 
#endif
#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_types.h"
#if 0
#line 161
struct cudaTextureDesc { 
#line 166
cudaTextureAddressMode addressMode[3]; 
#line 170
cudaTextureFilterMode filterMode; 
#line 174
cudaTextureReadMode readMode; 
#line 178
int sRGB; 
#line 182
float borderColor[4]; 
#line 186
int normalizedCoords; 
#line 190
unsigned maxAnisotropy; 
#line 194
cudaTextureFilterMode mipmapFilterMode; 
#line 198
float mipmapLevelBias; 
#line 202
float minMipmapLevelClamp; 
#line 206
float maxMipmapLevelClamp; 
#line 207
}; 
#endif
#line 212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_types.h"
#if 0
typedef unsigned __int64 
#line 212
cudaTextureObject_t; 
#endif
#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\library_types.h"
typedef 
#line 54
enum cudaDataType_t { 
#line 56
CUDA_R_16F = 2, 
#line 57
CUDA_C_16F = 6, 
#line 58
CUDA_R_32F = 0, 
#line 59
CUDA_C_32F = 4, 
#line 60
CUDA_R_64F = 1, 
#line 61
CUDA_C_64F = 5, 
#line 62
CUDA_R_8I = 3, 
#line 63
CUDA_C_8I = 7, 
#line 64
CUDA_R_8U, 
#line 65
CUDA_C_8U, 
#line 66
CUDA_R_32I, 
#line 67
CUDA_C_32I, 
#line 68
CUDA_R_32U, 
#line 69
CUDA_C_32U
#line 70
} cudaDataType; 
#line 78
typedef 
#line 73
enum libraryPropertyType_t { 
#line 75
MAJOR_VERSION, 
#line 76
MINOR_VERSION, 
#line 77
PATCH_LEVEL
#line 78
} libraryPropertyType; 
#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_device_runtime_api.h"
extern "C" {
#line 123
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 124
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 125
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 126
extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 127
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 128
extern cudaError_t __stdcall cudaGetLastError(); 
#line 129
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 130
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 131
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 132
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 133
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 134
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 135
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 136
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 137
extern cudaError_t __stdcall cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 138
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 139
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
#line 140
extern cudaError_t __stdcall cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
#line 141
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 142
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 143
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 144
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 145
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 146
extern cudaError_t __stdcall cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 147
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 148
extern cudaError_t __stdcall cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 149
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 150
extern cudaError_t __stdcall cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 151
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 152
extern cudaError_t __stdcall cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 153
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 154
extern cudaError_t __stdcall cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 155
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 156
extern cudaError_t __stdcall cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 157
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 178
extern void *__stdcall cudaGetParameterBuffer(size_t alignment, size_t size); 
#line 206
extern void *__stdcall cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
#line 207
extern cudaError_t __stdcall cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 208
extern cudaError_t __stdcall cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
#line 226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 227
extern cudaError_t __stdcall cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
#line 230 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
#line 231
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 233
extern unsigned __int64 __stdcall cudaCGGetIntrinsicHandle(cudaCGScope scope); 
#line 234
extern cudaError_t __stdcall cudaCGSynchronize(unsigned __int64 handle, unsigned flags); 
#line 235
extern cudaError_t __stdcall cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned __int64 handle); 
#line 236
extern cudaError_t __stdcall cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned __int64 handle); 
#line 237
}
#line 239
template< class T> static __inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
#line 240
template< class T> static __inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
#line 241
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
#line 242
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_runtime_api.h"
extern "C" {
#line 251 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceReset(); 
#line 270
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 347
extern cudaError_t __stdcall cudaDeviceSetLimit(cudaLimit limit, size_t value); 
#line 378
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 410
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 446
extern cudaError_t __stdcall cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
#line 489
extern cudaError_t __stdcall cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
#line 519
extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 562
extern cudaError_t __stdcall cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
#line 587
extern cudaError_t __stdcall cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
#line 615
extern cudaError_t __stdcall cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
#line 660
extern cudaError_t __stdcall cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
#line 698
extern cudaError_t __stdcall cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
#line 739
extern cudaError_t __stdcall cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
#line 792
extern cudaError_t __stdcall cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
#line 825
extern cudaError_t __stdcall cudaIpcCloseMemHandle(void * devPtr); 
#line 865
extern cudaError_t __stdcall cudaThreadExit(); 
#line 889
extern cudaError_t __stdcall cudaThreadSynchronize(); 
#line 936
extern cudaError_t __stdcall cudaThreadSetLimit(cudaLimit limit, size_t value); 
#line 967
extern cudaError_t __stdcall cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
#line 1002
extern cudaError_t __stdcall cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 1048
extern cudaError_t __stdcall cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
#line 1104
extern cudaError_t __stdcall cudaGetLastError(); 
#line 1147
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 1163
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 1179
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 1210
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 1476
extern cudaError_t __stdcall cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
#line 1663
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 1701
extern cudaError_t __stdcall cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
#line 1720
extern cudaError_t __stdcall cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
#line 1755
extern cudaError_t __stdcall cudaSetDevice(int device); 
#line 1774
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 1803
extern cudaError_t __stdcall cudaSetValidDevices(int * device_arr, int len); 
#line 1866
extern cudaError_t __stdcall cudaSetDeviceFlags(unsigned flags); 
#line 1909
extern cudaError_t __stdcall cudaGetDeviceFlags(unsigned * flags); 
#line 1947
extern cudaError_t __stdcall cudaStreamCreate(cudaStream_t * pStream); 
#line 1977
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 2021
extern cudaError_t __stdcall cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
#line 2046
extern cudaError_t __stdcall cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
#line 2069
extern cudaError_t __stdcall cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
#line 2098
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 2122
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 2136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_runtime_api.h"
typedef void (__stdcall *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
#line 2195
extern cudaError_t __stdcall cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
#line 2217
extern cudaError_t __stdcall cudaStreamSynchronize(cudaStream_t stream); 
#line 2240
extern cudaError_t __stdcall cudaStreamQuery(cudaStream_t stream); 
#line 2321
extern cudaError_t __stdcall cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
#line 2358
extern cudaError_t __stdcall cudaEventCreate(cudaEvent_t * event); 
#line 2394
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 2432
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
#line 2462
extern cudaError_t __stdcall cudaEventQuery(cudaEvent_t event); 
#line 2491
extern cudaError_t __stdcall cudaEventSynchronize(cudaEvent_t event); 
#line 2517
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 2559
extern cudaError_t __stdcall cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
#line 2622
extern cudaError_t __stdcall cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 2677
extern cudaError_t __stdcall cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 2774
extern cudaError_t __stdcall cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
#line 2823
extern cudaError_t __stdcall cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
#line 2878
extern cudaError_t __stdcall cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
#line 2913
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 2952
extern cudaError_t __stdcall cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
#line 2976
extern cudaError_t __stdcall cudaSetDoubleForDevice(double * d); 
#line 3000
extern cudaError_t __stdcall cudaSetDoubleForHost(double * d); 
#line 3055
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
#line 3099
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
#line 3149
extern cudaError_t __stdcall cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0); 
#line 3178
extern cudaError_t __stdcall cudaSetupArgument(const void * arg, size_t size, size_t offset); 
#line 3219
extern cudaError_t __stdcall cudaLaunch(const void * func); 
#line 3338
extern cudaError_t __stdcall cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
#line 3366
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 3397
extern cudaError_t __stdcall cudaMallocHost(void ** ptr, size_t size); 
#line 3438
extern cudaError_t __stdcall cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
#line 3482
extern cudaError_t __stdcall cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
#line 3510
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 3532
extern cudaError_t __stdcall cudaFreeHost(void * ptr); 
#line 3554
extern cudaError_t __stdcall cudaFreeArray(cudaArray_t array); 
#line 3576
extern cudaError_t __stdcall cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
#line 3640
extern cudaError_t __stdcall cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
#line 3722
extern cudaError_t __stdcall cudaHostRegister(void * ptr, size_t size, unsigned flags); 
#line 3743
extern cudaError_t __stdcall cudaHostUnregister(void * ptr); 
#line 3786
extern cudaError_t __stdcall cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
#line 3806
extern cudaError_t __stdcall cudaHostGetFlags(unsigned * pFlags, void * pHost); 
#line 3843
extern cudaError_t __stdcall cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
#line 3980
extern cudaError_t __stdcall cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
#line 4117
extern cudaError_t __stdcall cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
#line 4144
extern cudaError_t __stdcall cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
#line 4247
extern cudaError_t __stdcall cudaMemcpy3D(const cudaMemcpy3DParms * p); 
#line 4276
extern cudaError_t __stdcall cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
#line 4392
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
#line 4416
extern cudaError_t __stdcall cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
#line 4437
extern cudaError_t __stdcall cudaMemGetInfo(size_t * free, size_t * total); 
#line 4461
extern cudaError_t __stdcall cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
#line 4502
extern cudaError_t __stdcall cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 4535
extern cudaError_t __stdcall cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
#line 4574
extern cudaError_t __stdcall cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
#line 4612
extern cudaError_t __stdcall cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
#line 4651
extern cudaError_t __stdcall cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 4697
extern cudaError_t __stdcall cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 4744
extern cudaError_t __stdcall cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 4791
extern cudaError_t __stdcall cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
#line 4836
extern cudaError_t __stdcall cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 4877
extern cudaError_t __stdcall cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
#line 4918
extern cudaError_t __stdcall cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
#line 4972
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 5005
extern cudaError_t __stdcall cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
#line 5052
extern cudaError_t __stdcall cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 5098
extern cudaError_t __stdcall cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 5158
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 5213
extern cudaError_t __stdcall cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 5267
extern cudaError_t __stdcall cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 5316
extern cudaError_t __stdcall cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 5365
extern cudaError_t __stdcall cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 5392
extern cudaError_t __stdcall cudaMemset(void * devPtr, int value, size_t count); 
#line 5424
extern cudaError_t __stdcall cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
#line 5466
extern cudaError_t __stdcall cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
#line 5500
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
#line 5539
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
#line 5588
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
#line 5614
extern cudaError_t __stdcall cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
#line 5639
extern cudaError_t __stdcall cudaGetSymbolSize(size_t * size, const void * symbol); 
#line 5707
extern cudaError_t __stdcall cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
#line 5821
extern cudaError_t __stdcall cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
#line 5878
extern cudaError_t __stdcall cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
#line 5915
extern cudaError_t __stdcall cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
#line 6069
extern cudaError_t __stdcall cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
#line 6108
extern cudaError_t __stdcall cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
#line 6148
extern cudaError_t __stdcall cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
#line 6168
extern cudaError_t __stdcall cudaDeviceDisablePeerAccess(int peerDevice); 
#line 6229
extern cudaError_t __stdcall cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
#line 6262
extern cudaError_t __stdcall cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
#line 6299
extern cudaError_t __stdcall cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 6332
extern cudaError_t __stdcall cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 6362
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
#line 6398
extern cudaError_t __stdcall cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
#line 6425
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
#line 6465
extern cudaError_t __stdcall cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
#line 6501
extern cudaChannelFormatDesc __stdcall cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
#line 6552
extern cudaError_t __stdcall cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = 4294967295U); 
#line 6607
extern cudaError_t __stdcall cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
#line 6641
extern cudaError_t __stdcall cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
#line 6677
extern cudaError_t __stdcall cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
#line 6699
extern cudaError_t __stdcall cudaUnbindTexture(const textureReference * texref); 
#line 6724
extern cudaError_t __stdcall cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
#line 6750
extern cudaError_t __stdcall cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
#line 6791
extern cudaError_t __stdcall cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
#line 6812
extern cudaError_t __stdcall cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
#line 7040
extern cudaError_t __stdcall cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
#line 7057
extern cudaError_t __stdcall cudaDestroyTextureObject(cudaTextureObject_t texObject); 
#line 7075
extern cudaError_t __stdcall cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
#line 7093
extern cudaError_t __stdcall cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
#line 7112
extern cudaError_t __stdcall cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
#line 7153
extern cudaError_t __stdcall cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
#line 7170
extern cudaError_t __stdcall cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
#line 7187
extern cudaError_t __stdcall cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
#line 7216
extern cudaError_t __stdcall cudaDriverGetVersion(int * driverVersion); 
#line 7235
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 7240
extern cudaError_t __stdcall cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
#line 7477 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_runtime_api.h"
}
#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\channel_descriptor.h"
template< class T> __inline ::cudaChannelFormatDesc cudaCreateChannelDesc() 
#line 108
{ 
#line 109
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
#line 110
} 
#line 112
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
#line 113
{ 
#line 114
int e = (((int)sizeof(unsigned short)) * 8); 
#line 116
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 117
} 
#line 119
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
#line 120
{ 
#line 121
int e = (((int)sizeof(unsigned short)) * 8); 
#line 123
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 124
} 
#line 126
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
#line 127
{ 
#line 128
int e = (((int)sizeof(unsigned short)) * 8); 
#line 130
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 131
} 
#line 133
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
#line 134
{ 
#line 135
int e = (((int)sizeof(unsigned short)) * 8); 
#line 137
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 138
} 
#line 140
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
#line 141
{ 
#line 142
int e = (((int)sizeof(char)) * 8); 
#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\channel_descriptor.h"
} 
#line 151
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
#line 152
{ 
#line 153
int e = (((int)sizeof(signed char)) * 8); 
#line 155
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 156
} 
#line 158
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
#line 159
{ 
#line 160
int e = (((int)sizeof(unsigned char)) * 8); 
#line 162
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 163
} 
#line 165
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
#line 166
{ 
#line 167
int e = (((int)sizeof(signed char)) * 8); 
#line 169
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 170
} 
#line 172
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
#line 173
{ 
#line 174
int e = (((int)sizeof(unsigned char)) * 8); 
#line 176
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 177
} 
#line 179
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
#line 180
{ 
#line 181
int e = (((int)sizeof(signed char)) * 8); 
#line 183
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 184
} 
#line 186
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
#line 187
{ 
#line 188
int e = (((int)sizeof(unsigned char)) * 8); 
#line 190
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 191
} 
#line 193
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
#line 194
{ 
#line 195
int e = (((int)sizeof(signed char)) * 8); 
#line 197
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 198
} 
#line 200
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
#line 201
{ 
#line 202
int e = (((int)sizeof(unsigned char)) * 8); 
#line 204
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 205
} 
#line 207
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
#line 208
{ 
#line 209
int e = (((int)sizeof(short)) * 8); 
#line 211
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 212
} 
#line 214
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
#line 215
{ 
#line 216
int e = (((int)sizeof(unsigned short)) * 8); 
#line 218
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 219
} 
#line 221
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
#line 222
{ 
#line 223
int e = (((int)sizeof(short)) * 8); 
#line 225
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 226
} 
#line 228
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
#line 229
{ 
#line 230
int e = (((int)sizeof(unsigned short)) * 8); 
#line 232
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 233
} 
#line 235
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
#line 236
{ 
#line 237
int e = (((int)sizeof(short)) * 8); 
#line 239
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 240
} 
#line 242
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
#line 243
{ 
#line 244
int e = (((int)sizeof(unsigned short)) * 8); 
#line 246
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 247
} 
#line 249
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
#line 250
{ 
#line 251
int e = (((int)sizeof(short)) * 8); 
#line 253
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 254
} 
#line 256
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
#line 257
{ 
#line 258
int e = (((int)sizeof(unsigned short)) * 8); 
#line 260
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 261
} 
#line 263
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
#line 264
{ 
#line 265
int e = (((int)sizeof(int)) * 8); 
#line 267
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 268
} 
#line 270
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
#line 271
{ 
#line 272
int e = (((int)sizeof(unsigned)) * 8); 
#line 274
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 275
} 
#line 277
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
#line 278
{ 
#line 279
int e = (((int)sizeof(int)) * 8); 
#line 281
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 282
} 
#line 284
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
#line 285
{ 
#line 286
int e = (((int)sizeof(unsigned)) * 8); 
#line 288
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 289
} 
#line 291
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
#line 292
{ 
#line 293
int e = (((int)sizeof(int)) * 8); 
#line 295
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 296
} 
#line 298
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
#line 299
{ 
#line 300
int e = (((int)sizeof(unsigned)) * 8); 
#line 302
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 303
} 
#line 305
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
#line 306
{ 
#line 307
int e = (((int)sizeof(int)) * 8); 
#line 309
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 310
} 
#line 312
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
#line 313
{ 
#line 314
int e = (((int)sizeof(unsigned)) * 8); 
#line 316
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 317
} 
#line 321
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long> () 
#line 322
{ 
#line 323
int e = (((int)sizeof(long)) * 8); 
#line 325
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 326
} 
#line 328
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned long> () 
#line 329
{ 
#line 330
int e = (((int)sizeof(unsigned long)) * 8); 
#line 332
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 333
} 
#line 335
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long1> () 
#line 336
{ 
#line 337
int e = (((int)sizeof(long)) * 8); 
#line 339
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 340
} 
#line 342
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong1> () 
#line 343
{ 
#line 344
int e = (((int)sizeof(unsigned long)) * 8); 
#line 346
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 347
} 
#line 349
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long2> () 
#line 350
{ 
#line 351
int e = (((int)sizeof(long)) * 8); 
#line 353
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 354
} 
#line 356
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong2> () 
#line 357
{ 
#line 358
int e = (((int)sizeof(unsigned long)) * 8); 
#line 360
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 361
} 
#line 363
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long4> () 
#line 364
{ 
#line 365
int e = (((int)sizeof(long)) * 8); 
#line 367
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 368
} 
#line 370
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong4> () 
#line 371
{ 
#line 372
int e = (((int)sizeof(unsigned long)) * 8); 
#line 374
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 375
} 
#line 379 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\channel_descriptor.h"
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
#line 380
{ 
#line 381
int e = (((int)sizeof(float)) * 8); 
#line 383
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 384
} 
#line 386
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
#line 387
{ 
#line 388
int e = (((int)sizeof(float)) * 8); 
#line 390
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 391
} 
#line 393
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
#line 394
{ 
#line 395
int e = (((int)sizeof(float)) * 8); 
#line 397
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 398
} 
#line 400
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
#line 401
{ 
#line 402
int e = (((int)sizeof(float)) * 8); 
#line 404
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 405
} 
#line 79 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\driver_functions.h"
static __inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
#line 80
{ 
#line 81
cudaPitchedPtr s; 
#line 83
(s.ptr) = d; 
#line 84
(s.pitch) = p; 
#line 85
(s.xsize) = xsz; 
#line 86
(s.ysize) = ysz; 
#line 88
return s; 
#line 89
} 
#line 106
static __inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
#line 107
{ 
#line 108
cudaPos p; 
#line 110
(p.x) = x; 
#line 111
(p.y) = y; 
#line 112
(p.z) = z; 
#line 114
return p; 
#line 115
} 
#line 132
static __inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
#line 133
{ 
#line 134
cudaExtent e; 
#line 136
(e.width) = w; 
#line 137
(e.height) = h; 
#line 138
(e.depth) = d; 
#line 140
return e; 
#line 141
} 
#line 75 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_functions.h"
static __inline char1 make_char1(signed char x); 
#line 77
static __inline uchar1 make_uchar1(unsigned char x); 
#line 79
static __inline char2 make_char2(signed char x, signed char y); 
#line 81
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
#line 83
static __inline char3 make_char3(signed char x, signed char y, signed char z); 
#line 85
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
#line 87
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
#line 89
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
#line 91
static __inline short1 make_short1(short x); 
#line 93
static __inline ushort1 make_ushort1(unsigned short x); 
#line 95
static __inline short2 make_short2(short x, short y); 
#line 97
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
#line 99
static __inline short3 make_short3(short x, short y, short z); 
#line 101
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
#line 103
static __inline short4 make_short4(short x, short y, short z, short w); 
#line 105
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
#line 107
static __inline int1 make_int1(int x); 
#line 109
static __inline uint1 make_uint1(unsigned x); 
#line 111
static __inline int2 make_int2(int x, int y); 
#line 113
static __inline uint2 make_uint2(unsigned x, unsigned y); 
#line 115
static __inline int3 make_int3(int x, int y, int z); 
#line 117
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
#line 119
static __inline int4 make_int4(int x, int y, int z, int w); 
#line 121
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
#line 123
static __inline long1 make_long1(long x); 
#line 125
static __inline ulong1 make_ulong1(unsigned long x); 
#line 127
static __inline long2 make_long2(long x, long y); 
#line 129
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
#line 131
static __inline long3 make_long3(long x, long y, long z); 
#line 133
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
#line 135
static __inline long4 make_long4(long x, long y, long z, long w); 
#line 137
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
#line 139
static __inline float1 make_float1(float x); 
#line 141
static __inline float2 make_float2(float x, float y); 
#line 143
static __inline float3 make_float3(float x, float y, float z); 
#line 145
static __inline float4 make_float4(float x, float y, float z, float w); 
#line 147
static __inline longlong1 make_longlong1(__int64 x); 
#line 149
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x); 
#line 151
static __inline longlong2 make_longlong2(__int64 x, __int64 y); 
#line 153
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y); 
#line 155
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z); 
#line 157
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z); 
#line 159
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w); 
#line 161
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w); 
#line 163
static __inline double1 make_double1(double x); 
#line 165
static __inline double2 make_double2(double x, double y); 
#line 167
static __inline double3 make_double3(double x, double y, double z); 
#line 169
static __inline double4 make_double4(double x, double y, double z, double w); 
#line 75 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\vector_functions.hpp"
static __inline char1 make_char1(signed char x) 
#line 76
{ 
#line 77
char1 t; (t.x) = x; return t; 
#line 78
} 
#line 80
static __inline uchar1 make_uchar1(unsigned char x) 
#line 81
{ 
#line 82
uchar1 t; (t.x) = x; return t; 
#line 83
} 
#line 85
static __inline char2 make_char2(signed char x, signed char y) 
#line 86
{ 
#line 87
char2 t; (t.x) = x; (t.y) = y; return t; 
#line 88
} 
#line 90
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
#line 91
{ 
#line 92
uchar2 t; (t.x) = x; (t.y) = y; return t; 
#line 93
} 
#line 95
static __inline char3 make_char3(signed char x, signed char y, signed char z) 
#line 96
{ 
#line 97
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 98
} 
#line 100
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
#line 101
{ 
#line 102
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 103
} 
#line 105
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
#line 106
{ 
#line 107
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 108
} 
#line 110
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
#line 111
{ 
#line 112
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 113
} 
#line 115
static __inline short1 make_short1(short x) 
#line 116
{ 
#line 117
short1 t; (t.x) = x; return t; 
#line 118
} 
#line 120
static __inline ushort1 make_ushort1(unsigned short x) 
#line 121
{ 
#line 122
ushort1 t; (t.x) = x; return t; 
#line 123
} 
#line 125
static __inline short2 make_short2(short x, short y) 
#line 126
{ 
#line 127
short2 t; (t.x) = x; (t.y) = y; return t; 
#line 128
} 
#line 130
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
#line 131
{ 
#line 132
ushort2 t; (t.x) = x; (t.y) = y; return t; 
#line 133
} 
#line 135
static __inline short3 make_short3(short x, short y, short z) 
#line 136
{ 
#line 137
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 138
} 
#line 140
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
#line 141
{ 
#line 142
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 143
} 
#line 145
static __inline short4 make_short4(short x, short y, short z, short w) 
#line 146
{ 
#line 147
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 148
} 
#line 150
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
#line 151
{ 
#line 152
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 153
} 
#line 155
static __inline int1 make_int1(int x) 
#line 156
{ 
#line 157
int1 t; (t.x) = x; return t; 
#line 158
} 
#line 160
static __inline uint1 make_uint1(unsigned x) 
#line 161
{ 
#line 162
uint1 t; (t.x) = x; return t; 
#line 163
} 
#line 165
static __inline int2 make_int2(int x, int y) 
#line 166
{ 
#line 167
int2 t; (t.x) = x; (t.y) = y; return t; 
#line 168
} 
#line 170
static __inline uint2 make_uint2(unsigned x, unsigned y) 
#line 171
{ 
#line 172
uint2 t; (t.x) = x; (t.y) = y; return t; 
#line 173
} 
#line 175
static __inline int3 make_int3(int x, int y, int z) 
#line 176
{ 
#line 177
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 178
} 
#line 180
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
#line 181
{ 
#line 182
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 183
} 
#line 185
static __inline int4 make_int4(int x, int y, int z, int w) 
#line 186
{ 
#line 187
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 188
} 
#line 190
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
#line 191
{ 
#line 192
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 193
} 
#line 195
static __inline long1 make_long1(long x) 
#line 196
{ 
#line 197
long1 t; (t.x) = x; return t; 
#line 198
} 
#line 200
static __inline ulong1 make_ulong1(unsigned long x) 
#line 201
{ 
#line 202
ulong1 t; (t.x) = x; return t; 
#line 203
} 
#line 205
static __inline long2 make_long2(long x, long y) 
#line 206
{ 
#line 207
long2 t; (t.x) = x; (t.y) = y; return t; 
#line 208
} 
#line 210
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
#line 211
{ 
#line 212
ulong2 t; (t.x) = x; (t.y) = y; return t; 
#line 213
} 
#line 215
static __inline long3 make_long3(long x, long y, long z) 
#line 216
{ 
#line 217
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 218
} 
#line 220
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
#line 221
{ 
#line 222
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 223
} 
#line 225
static __inline long4 make_long4(long x, long y, long z, long w) 
#line 226
{ 
#line 227
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 228
} 
#line 230
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
#line 231
{ 
#line 232
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 233
} 
#line 235
static __inline float1 make_float1(float x) 
#line 236
{ 
#line 237
float1 t; (t.x) = x; return t; 
#line 238
} 
#line 240
static __inline float2 make_float2(float x, float y) 
#line 241
{ 
#line 242
float2 t; (t.x) = x; (t.y) = y; return t; 
#line 243
} 
#line 245
static __inline float3 make_float3(float x, float y, float z) 
#line 246
{ 
#line 247
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 248
} 
#line 250
static __inline float4 make_float4(float x, float y, float z, float w) 
#line 251
{ 
#line 252
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 253
} 
#line 255
static __inline longlong1 make_longlong1(__int64 x) 
#line 256
{ 
#line 257
longlong1 t; (t.x) = x; return t; 
#line 258
} 
#line 260
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x) 
#line 261
{ 
#line 262
ulonglong1 t; (t.x) = x; return t; 
#line 263
} 
#line 265
static __inline longlong2 make_longlong2(__int64 x, __int64 y) 
#line 266
{ 
#line 267
longlong2 t; (t.x) = x; (t.y) = y; return t; 
#line 268
} 
#line 270
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y) 
#line 271
{ 
#line 272
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
#line 273
} 
#line 275
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z) 
#line 276
{ 
#line 277
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 278
} 
#line 280
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z) 
#line 281
{ 
#line 282
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 283
} 
#line 285
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w) 
#line 286
{ 
#line 287
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 288
} 
#line 290
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w) 
#line 291
{ 
#line 292
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 293
} 
#line 295
static __inline double1 make_double1(double x) 
#line 296
{ 
#line 297
double1 t; (t.x) = x; return t; 
#line 298
} 
#line 300
static __inline double2 make_double2(double x, double y) 
#line 301
{ 
#line 302
double2 t; (t.x) = x; (t.y) = y; return t; 
#line 303
} 
#line 305
static __inline double3 make_double3(double x, double y, double z) 
#line 306
{ 
#line 307
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 308
} 
#line 310
static __inline double4 make_double4(double x, double y, double z, double w) 
#line 311
{ 
#line 312
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 313
} 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\errno.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
int *__cdecl _errno(); 
#line 22
errno_t __cdecl _set_errno(int _Value); 
#line 23
errno_t __cdecl _get_errno(int * _Value); 
#line 25
unsigned long *__cdecl __doserrno(); 
#line 28
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 29
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 130 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\errno.h"
}__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_string.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
const void *__cdecl memchr(const void * _Buf, int _Val, size_t _MaxCount); 
#line 26
int __cdecl memcmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 40 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_string.h"
void *__cdecl memcpy(void * _Dst, const void * _Src, size_t _Size); 
#line 47
void *__cdecl memmove(void * _Dst, const void * _Src, size_t _Size); 
#line 60 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_string.h"
void *__cdecl memset(void * _Dst, int _Val, size_t _Size); 
#line 67
const char *__cdecl strchr(const char * _Str, int _Val); 
#line 73
const char *__cdecl strrchr(const char * _Str, int _Ch); 
#line 79
const char *__cdecl strstr(const char * _Str, const char * _SubStr); 
#line 86
const __wchar_t *__cdecl wcschr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 92
const __wchar_t *__cdecl wcsrchr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 99
const __wchar_t *__cdecl wcsstr(const __wchar_t * _Str, const __wchar_t * _SubStr); 
#line 106
}__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_memcpy_s.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 35 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_memcpy_s.h"
static __inline errno_t __cdecl memcpy_s(void *const 
#line 36
_Destination, const rsize_t 
#line 37
_DestinationSize, const void *const 
#line 38
_Source, const rsize_t 
#line 39
_SourceSize) 
#line 41
{ 
#line 42
if (_SourceSize == (0)) 
#line 43
{ 
#line 44
return 0; 
#line 45
}  
#line 47
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 48
if ((_Source == (0)) || (_DestinationSize < _SourceSize)) 
#line 49
{ 
#line 50
memset(_Destination, 0, _DestinationSize); 
#line 52
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 53
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 56
return 22; 
#line 57
}  
#pragma warning(suppress:4996)
memcpy(_Destination, _Source, _SourceSize); 
#line 60
return 0; 
#line 61
} 
#line 64
static __inline errno_t __cdecl memmove_s(void *const 
#line 65
_Destination, const rsize_t 
#line 66
_DestinationSize, const void *const 
#line 67
_Source, const rsize_t 
#line 68
_SourceSize) 
#line 70
{ 
#line 71
if (_SourceSize == (0)) 
#line 72
{ 
#line 73
return 0; 
#line 74
}  
#line 76
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 77
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 78
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 80
#pragma warning(suppress:4996)
memmove(_Destination, _Source, _SourceSize); 
#line 82
return 0; 
#line 83
} 
#line 89 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_memcpy_s.h"
}__pragma( pack ( pop )) 
#line 19 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_memory.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 24
int __cdecl _memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 31
int __cdecl _memicmp_l(const void * _Buf1, const void * _Buf2, size_t _Size, _locale_t _Locale); 
#line 79 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_memory.h"
void *__cdecl memccpy(void * _Dst, const void * _Src, int _Val, size_t _Size); 
#line 87
int __cdecl memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_memory.h"
extern "C++" inline void *__cdecl memchr(void *
#line 101
_Pv, int 
#line 102
_C, size_t 
#line 103
_N) 
#line 105
{ 
#line 106
const void *const _Pvc = _Pv; 
#line 107
return const_cast< void *>(memchr(_Pvc, _C, _N)); 
#line 108
} 
#line 114 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_memory.h"
}__pragma( pack ( pop )) 
#line 16 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 28
errno_t __cdecl wcscat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 35
errno_t __cdecl wcscpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 42
errno_t __cdecl wcsncat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 50
errno_t __cdecl wcsncpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 58
__wchar_t *__cdecl wcstok_s(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 79 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__declspec(allocator) __wchar_t *__cdecl _wcsdup(const __wchar_t * _String); 
#line 89 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscat_s ( _Destination, _Size, _Source ); }}
#line 97 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
__wchar_t *__cdecl wcscat(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 105 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 109 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcscmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 114
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscpy_s ( _Destination, _Size, _Source ); }}
#line 120 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
__wchar_t *__cdecl wcscpy(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 128 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 131
size_t __cdecl wcscspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 137
size_t __cdecl wcslen(const __wchar_t * _String); 
#line 150 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsnlen(const __wchar_t * _Source, size_t _MaxCount); 
#line 166 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
static __inline size_t __cdecl wcsnlen_s(const __wchar_t *
#line 167
_Source, size_t 
#line 168
_MaxCount) 
#line 170
{ 
#line 171
return (_Source == (0)) ? 0 : wcsnlen(_Source, _MaxCount); 
#line 172
} 
#line 176 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 183 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncat(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 192 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsncmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 198
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 205 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncpy(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 214 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
const __wchar_t *__cdecl wcspbrk(const __wchar_t * _String, const __wchar_t * _Control); 
#line 220
size_t __cdecl wcsspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 226
__wchar_t *__cdecl wcstok(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 243 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 4141 4996)
#pragma warning(disable: 28719 28726 28727)
#line 247
static __inline __wchar_t *__cdecl _wcstok(__wchar_t *const 
#line 248
_String, const __wchar_t *const 
#line 249
_Delimiter) 
#line 251
{ 
#line 252
return wcstok(_String, _Delimiter, 0); 
#line 253
} 
#line 261 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
extern "C++" 
#line 260
__declspec(deprecated("wcstok has been changed to conform with the ISO C standard, adding an extra context parameter. To use the legacy Microsoft wcsto" "k, define _CRT_NON_CONFORMING_WCSTOK.")) inline __wchar_t *__cdecl 
#line 261
wcstok(__wchar_t *
#line 262
_String, const __wchar_t *
#line 263
_Delimiter) throw() 
#line 265
{ 
#line 266
return wcstok(_String, _Delimiter, 0); 
#line 267
} 
#line 270 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 278 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcserror(int _ErrorNumber); 
#line 283
errno_t __cdecl _wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, int _ErrorNumber); 
#line 289
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], int _Error ) throw ( ) { return _wcserror_s ( _Buffer, _Size, _Error ); }}
#line 298 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl __wcserror(const __wchar_t * _String); 
#line 302
errno_t __cdecl __wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _ErrorMessage); 
#line 308
extern "C++" {template < size_t _Size > inline errno_t __cdecl __wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _ErrorMessage ) throw ( ) { return __wcserror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 314 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
int __cdecl _wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 319
int __cdecl _wcsicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 325
int __cdecl _wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 331
int __cdecl _wcsnicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 338
errno_t __cdecl _wcsnset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value, size_t _MaxCount); 
#line 345
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsnset_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t _Value, size_t _MaxCount ) throw ( ) { return _wcsnset_s ( _Destination, _Size, _Value, _MaxCount ); }}
#line 352 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 360 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsrev(__wchar_t * _String); 
#line 364
errno_t __cdecl _wcsset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value); 
#line 370
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsset_s ( wchar_t ( & _String ) [ _Size ], wchar_t _Value ) throw ( ) { return _wcsset_s ( _String, _Size, _Value ); }}
#line 376 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 383 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s(__wchar_t * _String, size_t _SizeInWords); 
#line 388
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcslwr_s ( _String, _Size ); }}
#line 393 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr(__wchar_t * _String); 
#line 399 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s_l(__wchar_t * _String, size_t _SizeInWords, _locale_t _Locale); 
#line 405
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcslwr_s_l ( _String, _Size, _Locale ); }}
#line 411 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr_l(__wchar_t * _String, _locale_t _Locale); 
#line 419 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s(__wchar_t * _String, size_t _Size); 
#line 424
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcsupr_s ( _String, _Size ); }}
#line 429 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr(__wchar_t * _String); 
#line 435 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s_l(__wchar_t * _String, size_t _Size, _locale_t _Locale); 
#line 441
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcsupr_s_l ( _String, _Size, _Locale ); }}
#line 447 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr_l(__wchar_t * _String, _locale_t _Locale); 
#line 456 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsxfrm(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount); 
#line 464
size_t __cdecl _wcsxfrm_l(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 472
int __cdecl wcscoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 478
int __cdecl _wcscoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 485
int __cdecl _wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 491
int __cdecl _wcsicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 498
int __cdecl _wcsncoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 505
int __cdecl _wcsncoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 513
int __cdecl _wcsnicoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 520
int __cdecl _wcsnicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 535
extern "C++" {
#line 539
inline __wchar_t *__cdecl wcschr(__wchar_t *_String, __wchar_t _C) 
#line 540
{ 
#line 541
return const_cast< __wchar_t *>(wcschr(static_cast< const __wchar_t *>(_String), _C)); 
#line 542
} 
#line 545
inline __wchar_t *__cdecl wcspbrk(__wchar_t *_String, const __wchar_t *_Control) 
#line 546
{ 
#line 547
return const_cast< __wchar_t *>(wcspbrk(static_cast< const __wchar_t *>(_String), _Control)); 
#line 548
} 
#line 551
inline __wchar_t *__cdecl wcsrchr(__wchar_t *_String, __wchar_t _C) 
#line 552
{ 
#line 553
return const_cast< __wchar_t *>(wcsrchr(static_cast< const __wchar_t *>(_String), _C)); 
#line 554
} 
#line 558
inline __wchar_t *__cdecl wcsstr(__wchar_t *_String, const __wchar_t *_SubStr) 
#line 559
{ 
#line 560
return const_cast< __wchar_t *>(wcsstr(static_cast< const __wchar_t *>(_String), _SubStr)); 
#line 561
} 
#line 563
}
#line 580 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsdup(const __wchar_t * _String); 
#line 592 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 598
int __cdecl wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 606
__wchar_t *__cdecl wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 614
__wchar_t *__cdecl wcsrev(__wchar_t * _String); 
#line 620
__wchar_t *__cdecl wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 627
__wchar_t *__cdecl wcslwr(__wchar_t * _String); 
#line 633
__wchar_t *__cdecl wcsupr(__wchar_t * _String); 
#line 638
int __cdecl wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 647 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstring.h"
}__pragma( pack ( pop )) 
#line 19 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 28
errno_t __cdecl strcpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 35
errno_t __cdecl strcat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 42
errno_t __cdecl strerror_s(char * _Buffer, size_t _SizeInBytes, int _ErrorNumber); 
#line 48
errno_t __cdecl strncat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 56
errno_t __cdecl strncpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 64
char *__cdecl strtok_s(char * _String, const char * _Delimiter, char ** _Context); 
#line 72 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
void *__cdecl _memccpy(void * _Dst, const void * _Src, int _Val, size_t _MaxCount); 
#line 79
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcat_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcat_s ( _Destination, _Size, _Source ); }}
#line 87 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
char *__cdecl strcat(char * _Destination, const char * _Source); 
#line 95 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
#pragma warning(pop)
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
int __cdecl strcmp(const char * _Str1, const char * _Str2); 
#line 106
int __cdecl _strcmpi(const char * _String1, const char * _String2); 
#line 112
int __cdecl strcoll(const char * _String1, const char * _String2); 
#line 118
int __cdecl _strcoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 124
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcpy_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcpy_s ( _Destination, _Size, _Source ); }}
#line 130 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
char *__cdecl strcpy(char * _Destination, const char * _Source); 
#line 138 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
#pragma warning(pop)
#line 141
size_t __cdecl strcspn(const char * _Str, const char * _Control); 
#line 152 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
__declspec(allocator) char *__cdecl _strdup(const char * _Source); 
#line 163 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl _strerror(const char * _ErrorMessage); 
#line 168
errno_t __cdecl _strerror_s(char * _Buffer, size_t _SizeInBytes, const char * _ErrorMessage); 
#line 174
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strerror_s ( char ( & _Buffer ) [ _Size ], char const * _ErrorMessage ) throw ( ) { return _strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 182 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl strerror(int _ErrorMessage); 
#line 186
extern "C++" {template < size_t _Size > inline errno_t __cdecl strerror_s ( char ( & _Buffer ) [ _Size ], int _ErrorMessage ) throw ( ) { return strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 193 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
int __cdecl _stricmp(const char * _String1, const char * _String2); 
#line 199
int __cdecl _stricoll(const char * _String1, const char * _String2); 
#line 205
int __cdecl _stricoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 212
int __cdecl _stricmp_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 219
size_t __cdecl strlen(const char * _Str); 
#line 224
errno_t __cdecl _strlwr_s(char * _String, size_t _Size); 
#line 229
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strlwr_s ( _String, _Size ); }}
#line 234 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl _strlwr(char * _String); 
#line 240 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
errno_t __cdecl _strlwr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 246
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strlwr_s_l ( _String, _Size, _Locale ); }}
#line 252 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl _strlwr_l(char * _String, _locale_t _Locale); 
#line 259 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncat_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 266 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl strncat(char * _Destination, const char * _Source, size_t _Count); 
#line 275 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
int __cdecl strncmp(const char * _Str1, const char * _Str2, size_t _MaxCount); 
#line 282
int __cdecl _strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 289
int __cdecl _strnicmp_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 297
int __cdecl _strnicoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 304
int __cdecl _strnicoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 312
int __cdecl _strncoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 319
int __cdecl _strncoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 326
size_t __cdecl __strncnt(const char * _String, size_t _Count); 
#line 331
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncpy_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 338 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl strncpy(char * _Destination, const char * _Source, size_t _Count); 
#line 355 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
size_t __cdecl strnlen(const char * _String, size_t _MaxCount); 
#line 371 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
static __inline size_t __cdecl strnlen_s(const char *
#line 372
_String, size_t 
#line 373
_MaxCount) 
#line 375
{ 
#line 376
return (_String == (0)) ? 0 : strnlen(_String, _MaxCount); 
#line 377
} 
#line 382 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
errno_t __cdecl _strnset_s(char * _String, size_t _SizeInBytes, int _Value, size_t _MaxCount); 
#line 389
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strnset_s ( char ( & _Destination ) [ _Size ], int _Value, size_t _Count ) throw ( ) { return _strnset_s ( _Destination, _Size, _Value, _Count ); }}
#line 396 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl _strnset(char * _Destination, int _Value, size_t _Count); 
#line 405 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
const char *__cdecl strpbrk(const char * _Str, const char * _Control); 
#line 410
char *__cdecl _strrev(char * _Str); 
#line 415
errno_t __cdecl _strset_s(char * _Destination, size_t _DestinationSize, int _Value); 
#line 421
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strset_s ( char ( & _Destination ) [ _Size ], int _Value ) throw ( ) { return _strset_s ( _Destination, _Size, _Value ); }}
#line 427 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl _strset(char * _Destination, int _Value); 
#line 434 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
size_t __cdecl strspn(const char * _Str, const char * _Control); 
#line 440
char *__cdecl strtok(char * _String, const char * _Delimiter); 
#line 446
errno_t __cdecl _strupr_s(char * _String, size_t _Size); 
#line 451
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strupr_s ( _String, _Size ); }}
#line 456 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl _strupr(char * _String); 
#line 462 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
errno_t __cdecl _strupr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 468
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strupr_s_l ( _String, _Size, _Locale ); }}
#line 474 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl _strupr_l(char * _String, _locale_t _Locale); 
#line 483 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
size_t __cdecl strxfrm(char * _Destination, const char * _Source, size_t _MaxCount); 
#line 491
size_t __cdecl _strxfrm_l(char * _Destination, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 501
extern "C++" {
#line 504
inline char *__cdecl strchr(char *const _String, const int _Ch) 
#line 505
{ 
#line 506
return const_cast< char *>(strchr(static_cast< const char *>(_String), _Ch)); 
#line 507
} 
#line 510
inline char *__cdecl strpbrk(char *const _String, const char *const _Control) 
#line 511
{ 
#line 512
return const_cast< char *>(strpbrk(static_cast< const char *>(_String), _Control)); 
#line 513
} 
#line 516
inline char *__cdecl strrchr(char *const _String, const int _Ch) 
#line 517
{ 
#line 518
return const_cast< char *>(strrchr(static_cast< const char *>(_String), _Ch)); 
#line 519
} 
#line 522
inline char *__cdecl strstr(char *const _String, const char *const _SubString) 
#line 523
{ 
#line 524
return const_cast< char *>(strstr(static_cast< const char *>(_String), _SubString)); 
#line 525
} 
#line 526
}
#line 536 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
char *__cdecl strdup(const char * _String); 
#line 543
int __cdecl strcmpi(const char * _String1, const char * _String2); 
#line 549
int __cdecl stricmp(const char * _String1, const char * _String2); 
#line 555
char *__cdecl strlwr(char * _String); 
#line 560
int __cdecl strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 567
char *__cdecl strnset(char * _String, int _Value, size_t _MaxCount); 
#line 574
char *__cdecl strrev(char * _String); 
#line 579
char *__cdecl strset(char * _String, int _Value); 
#line 584
char *__cdecl strupr(char * _String); 
#line 592 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\string.h"
}__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 22
struct tm { 
#line 24
int tm_sec; 
#line 25
int tm_min; 
#line 26
int tm_hour; 
#line 27
int tm_mday; 
#line 28
int tm_mon; 
#line 29
int tm_year; 
#line 30
int tm_wday; 
#line 31
int tm_yday; 
#line 32
int tm_isdst; 
#line 33
}; 
#line 44
__wchar_t *__cdecl _wasctime(const tm * _Tm); 
#line 50
errno_t __cdecl _wasctime_s(__wchar_t * _Buffer, size_t _SizeInWords, const tm * _Tm); 
#line 56
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wasctime_s ( wchar_t ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return _wasctime_s ( _Buffer, _Size, _Time ); }}
#line 65 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
size_t __cdecl wcsftime(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm); 
#line 74
size_t __cdecl _wcsftime_l(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm, _locale_t _Locale); 
#line 84
__wchar_t *__cdecl _wctime32(const __time32_t * _Time); 
#line 89
errno_t __cdecl _wctime32_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time32_t * _Time); 
#line 95
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime32_s ( wchar_t ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _wctime32_s ( _Buffer, _Size, _Time ); }}
#line 104 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wctime64(const __time64_t * _Time); 
#line 109
errno_t __cdecl _wctime64_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time64_t * _Time); 
#line 114
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime64_s ( wchar_t ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _wctime64_s ( _Buffer, _Size, _Time ); }}
#line 121 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrdate_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 126
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrdate_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrdate_s ( _Buffer, _Size ); }}
#line 131 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrdate(__wchar_t * _Buffer); 
#line 137 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrtime_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 142
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrtime_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrtime_s ( _Buffer, _Size ); }}
#line 147 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrtime(__wchar_t * _Buffer); 
#line 160 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
#pragma warning(push)
#pragma warning(disable: 4996)
#line 185
static __inline __wchar_t *__cdecl _wctime(const time_t *const 
#line 186
_Time) 
#line 187
{ 
#line 188
return _wctime64(_Time); 
#line 189
} 
#line 192
static __inline errno_t __cdecl _wctime_s(__wchar_t *const 
#line 193
_Buffer, const size_t 
#line 194
_SizeInWords, const time_t *const 
#line 195
_Time) 
#line 197
{ 
#line 198
return _wctime64_s(_Buffer, _SizeInWords, _Time); 
#line 199
} 
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
}
#line 203 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
#pragma warning(pop)
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wtime.h"
__pragma( pack ( pop )) 
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 24
typedef long clock_t; 
#line 26
struct _timespec32 { 
#line 28
__time32_t tv_sec; 
#line 29
long tv_nsec; 
#line 30
}; 
#line 32
struct _timespec64 { 
#line 34
__time64_t tv_sec; 
#line 35
long tv_nsec; 
#line 36
}; 
#line 39
struct timespec { 
#line 41
time_t tv_sec; 
#line 42
long tv_nsec; 
#line 43
}; 
#line 62 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
int *__cdecl __daylight(); 
#line 68
long *__cdecl __dstbias(); 
#line 74
long *__cdecl __timezone(); 
#line 80
char **__cdecl __tzname(); 
#line 85
errno_t __cdecl _get_daylight(int * _Daylight); 
#line 90
errno_t __cdecl _get_dstbias(long * _DaylightSavingsBias); 
#line 95
errno_t __cdecl _get_timezone(long * _TimeZone); 
#line 100
errno_t __cdecl _get_tzname(size_t * _ReturnValue, char * _Buffer, size_t _SizeInBytes, int _Index); 
#line 117
char *__cdecl asctime(const tm * _Tm); 
#line 124
errno_t __cdecl asctime_s(char * _Buffer, size_t _SizeInBytes, const tm * _Tm); 
#line 131 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl asctime_s ( char ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return asctime_s ( _Buffer, _Size, _Time ); }}
#line 138 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
clock_t __cdecl clock(); 
#line 143
char *__cdecl _ctime32(const __time32_t * _Time); 
#line 148
errno_t __cdecl _ctime32_s(char * _Buffer, size_t _SizeInBytes, const __time32_t * _Time); 
#line 154
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime32_s ( char ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _ctime32_s ( _Buffer, _Size, _Time ); }}
#line 163 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
char *__cdecl _ctime64(const __time64_t * _Time); 
#line 168
errno_t __cdecl _ctime64_s(char * _Buffer, size_t _SizeInBytes, const __time64_t * _Time); 
#line 174
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime64_s ( char ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _ctime64_s ( _Buffer, _Size, _Time ); }}
#line 181 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
double __cdecl _difftime32(__time32_t _Time1, __time32_t _Time2); 
#line 187
double __cdecl _difftime64(__time64_t _Time1, __time64_t _Time2); 
#line 194
tm *__cdecl _gmtime32(const __time32_t * _Time); 
#line 199
errno_t __cdecl _gmtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 206
tm *__cdecl _gmtime64(const __time64_t * _Time); 
#line 211
errno_t __cdecl _gmtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 218
tm *__cdecl _localtime32(const __time32_t * _Time); 
#line 223
errno_t __cdecl _localtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 230
tm *__cdecl _localtime64(const __time64_t * _Time); 
#line 235
errno_t __cdecl _localtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 241
__time32_t __cdecl _mkgmtime32(tm * _Tm); 
#line 246
__time64_t __cdecl _mkgmtime64(tm * _Tm); 
#line 251
__time32_t __cdecl _mktime32(tm * _Tm); 
#line 256
__time64_t __cdecl _mktime64(tm * _Tm); 
#line 262
size_t __cdecl strftime(char * _Buffer, size_t _SizeInBytes, const char * _Format, const tm * _Tm); 
#line 271
size_t __cdecl _strftime_l(char * _Buffer, size_t _MaxSize, const char * _Format, const tm * _Tm, _locale_t _Locale); 
#line 280
errno_t __cdecl _strdate_s(char * _Buffer, size_t _SizeInBytes); 
#line 285
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strdate_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strdate_s ( _Buffer, _Size ); }}
#line 290 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
char *__cdecl _strdate(char * _Buffer); 
#line 296 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
errno_t __cdecl _strtime_s(char * _Buffer, size_t _SizeInBytes); 
#line 301
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strtime_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strtime_s ( _Buffer, _Size ); }}
#line 306 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
char *__cdecl _strtime(char * _Buffer); 
#line 311 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
__time32_t __cdecl _time32(__time32_t * _Time); 
#line 315
__time64_t __cdecl _time64(__time64_t * _Time); 
#line 321
int __cdecl _timespec32_get(_timespec32 * _Ts, int _Base); 
#line 328
int __cdecl _timespec64_get(_timespec64 * _Ts, int _Base); 
#line 342
void __cdecl _tzset(); 
#line 345
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using GetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 346
_getsystime(tm * _Tm); 
#line 350
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 351
_setsystime(tm * _Tm, unsigned _Milliseconds); 
#line 476 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
static __inline char *__cdecl ctime(const time_t *const 
#line 477
_Time) 
#line 479
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _ctime64(_Time); 
#pragma warning(pop)
} 
#line 487
static __inline double __cdecl difftime(const time_t 
#line 488
_Time1, const time_t 
#line 489
_Time2) 
#line 491
{ 
#line 492
return _difftime64(_Time1, _Time2); 
#line 493
} 
#line 496
static __inline tm *__cdecl gmtime(const time_t *const 
#line 497
_Time) 
#line 498
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _gmtime64(_Time); 
#pragma warning(pop)
} 
#line 506
static __inline tm *__cdecl localtime(const time_t *const 
#line 507
_Time) 
#line 509
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _localtime64(_Time); 
#pragma warning(pop)
} 
#line 517
static __inline time_t __cdecl _mkgmtime(tm *const 
#line 518
_Tm) 
#line 520
{ 
#line 521
return _mkgmtime64(_Tm); 
#line 522
} 
#line 525
static __inline time_t __cdecl mktime(tm *const 
#line 526
_Tm) 
#line 528
{ 
#line 529
return _mktime64(_Tm); 
#line 530
} 
#line 532
static __inline time_t __cdecl time(time_t *const 
#line 533
_Time) 
#line 535
{ 
#line 536
return _time64(_Time); 
#line 537
} 
#line 540
static __inline int __cdecl timespec_get(timespec *const 
#line 541
_Ts, const int 
#line 542
_Base) 
#line 544
{ 
#line 545
return _timespec64_get((_timespec64 *)_Ts, _Base); 
#line 546
} 
#line 550
static __inline errno_t __cdecl ctime_s(char *const 
#line 551
_Buffer, const size_t 
#line 552
_SizeInBytes, const time_t *const 
#line 553
_Time) 
#line 555
{ 
#line 556
return _ctime64_s(_Buffer, _SizeInBytes, _Time); 
#line 557
} 
#line 560
static __inline errno_t __cdecl gmtime_s(tm *const 
#line 561
_Tm, const time_t *const 
#line 562
_Time) 
#line 564
{ 
#line 565
return _gmtime64_s(_Tm, _Time); 
#line 566
} 
#line 569
static __inline errno_t __cdecl localtime_s(tm *const 
#line 570
_Tm, const time_t *const 
#line 571
_Time) 
#line 573
{ 
#line 574
return _localtime64_s(_Tm, _Time); 
#line 575
} 
#line 594 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
void __cdecl tzset(); 
#line 601 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\time.h"
}__pragma( pack ( pop )) 
#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt/common_functions.h"
extern "C" {
#line 73 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt/common_functions.h"
extern clock_t __cdecl clock(); 
#line 78 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt/common_functions.h"
extern void *__cdecl memset(void *, int, size_t); 
#line 79
extern void *__cdecl memcpy(void *, const void *, size_t); 
#line 81
}
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern "C" {
#line 182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern int __cdecl abs(int); 
#line 183
extern long __cdecl labs(long); 
#line 184
extern __int64 llabs(__int64); 
#line 234 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl fabs(double x); 
#line 275
extern __inline float fabsf(float x); 
#line 279 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline int min(int, int); 
#line 281
extern inline unsigned umin(unsigned, unsigned); 
#line 282
extern inline __int64 llmin(__int64, __int64); 
#line 283
extern inline unsigned __int64 ullmin(unsigned __int64, unsigned __int64); 
#line 306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl fminf(float x, float y); 
#line 326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl fmin(double x, double y); 
#line 331 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline int max(int, int); 
#line 333
extern inline unsigned umax(unsigned, unsigned); 
#line 334
extern inline __int64 llmax(__int64, __int64); 
#line 335
extern inline unsigned __int64 ullmax(unsigned __int64, unsigned __int64); 
#line 358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl fmaxf(float x, float y); 
#line 378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl fmax(double, double); 
#line 420 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl sin(double x); 
#line 453
extern double __cdecl cos(double x); 
#line 472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern void sincos(double x, double * sptr, double * cptr); 
#line 488
extern void sincosf(float x, float * sptr, float * cptr); 
#line 533 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl tan(double x); 
#line 602
extern double __cdecl sqrt(double x); 
#line 674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double rsqrt(double x); 
#line 744
extern float rsqrtf(float x); 
#line 802 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl log2(double x); 
#line 827 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl exp2(double x); 
#line 852 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl exp2f(float x); 
#line 877 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double exp10(double x); 
#line 900
extern float exp10f(float x); 
#line 948 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl expm1(double x); 
#line 993 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl expm1f(float x); 
#line 1048 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl log2f(float x); 
#line 1100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl log10(double x); 
#line 1171
extern double __cdecl log(double x); 
#line 1267 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl log1p(double x); 
#line 1364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl log1pf(float x); 
#line 1437 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl floor(double x); 
#line 1476
extern double __cdecl exp(double x); 
#line 1507
extern double __cdecl cosh(double x); 
#line 1537
extern double __cdecl sinh(double x); 
#line 1567
extern double __cdecl tanh(double x); 
#line 1604 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl acosh(double x); 
#line 1642 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl acoshf(float x); 
#line 1658 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl asinh(double x); 
#line 1674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl asinhf(float x); 
#line 1728 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl atanh(double x); 
#line 1782 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl atanhf(float x); 
#line 1839 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl ldexp(double x, int exp); 
#line 1895
extern __inline float ldexpf(float x, int exp); 
#line 1949 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl logb(double x); 
#line 2004 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl logbf(float x); 
#line 2034 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern int __cdecl ilogb(double x); 
#line 2064 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern int __cdecl ilogbf(float x); 
#line 2140 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl scalbn(double x, int n); 
#line 2216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl scalbnf(float x, int n); 
#line 2292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl scalbln(double x, long n); 
#line 2368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl scalblnf(float x, long n); 
#line 2444 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl frexp(double x, int * nptr); 
#line 2519
extern __inline float frexpf(float x, int * nptr); 
#line 2535 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl round(double x); 
#line 2552 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl roundf(float x); 
#line 2570 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern long __cdecl lround(double x); 
#line 2588 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern long __cdecl lroundf(float x); 
#line 2606 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern __int64 __cdecl llround(double x); 
#line 2624 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern __int64 __cdecl llroundf(float x); 
#line 2676 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl rintf(float x); 
#line 2693 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern long __cdecl lrint(double x); 
#line 2710 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern long __cdecl lrintf(float x); 
#line 2727 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrint(double x); 
#line 2744 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrintf(float x); 
#line 2797 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl nearbyint(double x); 
#line 2850 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl nearbyintf(float x); 
#line 2910 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl ceil(double x); 
#line 2924 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl trunc(double x); 
#line 2939 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl truncf(float x); 
#line 2965 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl fdim(double x, double y); 
#line 2991 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl fdimf(float x, float y); 
#line 3025 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl atan2(double y, double x); 
#line 3056
extern double __cdecl atan(double x); 
#line 3079
extern double __cdecl acos(double x); 
#line 3111
extern double __cdecl asin(double x); 
#line 3154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl hypot(double x, double y); 
#line 3209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double rhypot(double x, double y); 
#line 3253 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline float __cdecl hypotf(float x, float y); 
#line 3307 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float rhypotf(float x, float y); 
#line 3351
extern double __cdecl norm3d(double a, double b, double c); 
#line 3402
extern double rnorm3d(double a, double b, double c); 
#line 3451
extern double __cdecl norm4d(double a, double b, double c, double d); 
#line 3507
extern double rnorm4d(double a, double b, double c, double d); 
#line 3552
extern double norm(int dim, const double * t); 
#line 3603
extern double rnorm(int dim, const double * t); 
#line 3655
extern float rnormf(int dim, const float * a); 
#line 3699
extern float normf(int dim, const float * a); 
#line 3744
extern float norm3df(float a, float b, float c); 
#line 3795
extern float rnorm3df(float a, float b, float c); 
#line 3844
extern float norm4df(float a, float b, float c, float d); 
#line 3900
extern float rnorm4df(float a, float b, float c, float d); 
#line 3989 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl cbrt(double x); 
#line 4075 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl cbrtf(float x); 
#line 4128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double rcbrt(double x); 
#line 4178
extern float rcbrtf(float x); 
#line 4238
extern double sinpi(double x); 
#line 4298
extern float sinpif(float x); 
#line 4350
extern double cospi(double x); 
#line 4402
extern float cospif(float x); 
#line 4432
extern void sincospi(double x, double * sptr, double * cptr); 
#line 4462
extern void sincospif(float x, float * sptr, float * cptr); 
#line 4774 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl pow(double x, double y); 
#line 4830
extern double __cdecl modf(double x, double * iptr); 
#line 4889
extern double __cdecl fmod(double x, double y); 
#line 4977 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl remainder(double x, double y); 
#line 5067 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl remainderf(float x, float y); 
#line 5121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl remquo(double x, double y, int * quo); 
#line 5175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl remquof(float x, float y, int * quo); 
#line 5214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl j0(double x); 
#line 5256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float j0f(float x); 
#line 5317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl j1(double x); 
#line 5378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float j1f(float x); 
#line 5421 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl jn(int n, double x); 
#line 5464 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float jnf(int n, float x); 
#line 5516 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl y0(double x); 
#line 5568 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float y0f(float x); 
#line 5620 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl y1(double x); 
#line 5672 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float y1f(float x); 
#line 5725 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl yn(int n, double x); 
#line 5778 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float ynf(int n, float x); 
#line 5805
extern double __cdecl cyl_bessel_i0(double x); 
#line 5831
extern float cyl_bessel_i0f(float x); 
#line 5858
extern double __cdecl cyl_bessel_i1(double x); 
#line 5884
extern float cyl_bessel_i1f(float x); 
#line 5969 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl erf(double x); 
#line 6051 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl erff(float x); 
#line 6113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double erfinv(double y); 
#line 6170
extern float erfinvf(float y); 
#line 6211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl erfc(double x); 
#line 6249 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl erfcf(float x); 
#line 6377 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl lgamma(double x); 
#line 6438 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double erfcinv(double y); 
#line 6494
extern float erfcinvf(float y); 
#line 6552
extern double normcdfinv(double y); 
#line 6610
extern float normcdfinvf(float y); 
#line 6653
extern double normcdf(double y); 
#line 6696
extern float normcdff(float y); 
#line 6771
extern double erfcx(double x); 
#line 6846
extern float erfcxf(float x); 
#line 6982 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl lgammaf(float x); 
#line 7091 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl tgamma(double x); 
#line 7200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl tgammaf(float x); 
#line 7213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl copysign(double x, double y); 
#line 7226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl copysignf(float x, float y); 
#line 7263 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl nextafter(double x, double y); 
#line 7300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl nextafterf(float x, float y); 
#line 7316 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl nan(const char * tagp); 
#line 7332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl nanf(const char * tagp); 
#line 7337 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern int __isinff(float); 
#line 7338
extern int __isnanf(float); 
#line 7348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern int __finite(double); 
#line 7349
extern int __finitef(float); 
#line 7350
extern int __signbit(double); 
#line 7351
extern int __isnan(double); 
#line 7352
extern int __isinf(double); 
#line 7355 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern int __signbitf(float); 
#line 7516 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern double __cdecl fma(double x, double y, double z); 
#line 7674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl fmaf(float x, float y, float z); 
#line 7683 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern int __signbitl(long double); 
#line 7689 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern int __finitel(long double); 
#line 7690
extern int __isinfl(long double); 
#line 7691
extern int __isnanl(long double); 
#line 7695 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern float __cdecl acosf(float); 
#line 7696
extern float __cdecl asinf(float); 
#line 7697
extern float __cdecl atanf(float); 
#line 7698
extern float __cdecl atan2f(float, float); 
#line 7699
extern float __cdecl cosf(float); 
#line 7700
extern float __cdecl sinf(float); 
#line 7701
extern float __cdecl tanf(float); 
#line 7702
extern float __cdecl coshf(float); 
#line 7703
extern float __cdecl sinhf(float); 
#line 7704
extern float __cdecl tanhf(float); 
#line 7705
extern float __cdecl expf(float); 
#line 7706
extern float __cdecl logf(float); 
#line 7707
extern float __cdecl log10f(float); 
#line 7708
extern float __cdecl modff(float, float *); 
#line 7709
extern float __cdecl powf(float, float); 
#line 7710
extern float __cdecl sqrtf(float); 
#line 7711
extern float __cdecl ceilf(float); 
#line 7712
extern float __cdecl floorf(float); 
#line 7713
extern float __cdecl fmodf(float, float); 
#line 8846 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
}
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 16
#pragma warning(push)
#pragma warning(disable:4738)
#pragma warning(disable:4820)
#line 25
struct _exception { 
#line 27
int type; 
#line 28
char *name; 
#line 29
double arg1; 
#line 30
double arg2; 
#line 31
double retval; 
#line 32
}; 
#line 39
struct _complex { 
#line 41
double x, y; 
#line 42
}; 
#line 61 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
typedef float float_t; 
#line 62
typedef double double_t; 
#line 80 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
extern const double _HUGE; 
#line 171 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
void __cdecl _fperrraise(int _Except); 
#line 173
short __cdecl _dclass(double _X); 
#line 174
short __cdecl _ldclass(long double _X); 
#line 175
short __cdecl _fdclass(float _X); 
#line 177
int __cdecl _dsign(double _X); 
#line 178
int __cdecl _ldsign(long double _X); 
#line 179
int __cdecl _fdsign(float _X); 
#line 181
int __cdecl _dpcomp(double _X, double _Y); 
#line 182
int __cdecl _ldpcomp(long double _X, long double _Y); 
#line 183
int __cdecl _fdpcomp(float _X, float _Y); 
#line 185
short __cdecl _dtest(double * _Px); 
#line 186
short __cdecl _ldtest(long double * _Px); 
#line 187
short __cdecl _fdtest(float * _Px); 
#line 189
short __cdecl _d_int(double * _Px, short _Xexp); 
#line 190
short __cdecl _ld_int(long double * _Px, short _Xexp); 
#line 191
short __cdecl _fd_int(float * _Px, short _Xexp); 
#line 193
short __cdecl _dscale(double * _Px, long _Lexp); 
#line 194
short __cdecl _ldscale(long double * _Px, long _Lexp); 
#line 195
short __cdecl _fdscale(float * _Px, long _Lexp); 
#line 197
short __cdecl _dunscale(short * _Pex, double * _Px); 
#line 198
short __cdecl _ldunscale(short * _Pex, long double * _Px); 
#line 199
short __cdecl _fdunscale(short * _Pex, float * _Px); 
#line 201
short __cdecl _dexp(double * _Px, double _Y, long _Eoff); 
#line 202
short __cdecl _ldexp(long double * _Px, long double _Y, long _Eoff); 
#line 203
short __cdecl _fdexp(float * _Px, float _Y, long _Eoff); 
#line 205
short __cdecl _dnorm(unsigned short * _Ps); 
#line 206
short __cdecl _fdnorm(unsigned short * _Ps); 
#line 208
double __cdecl _dpoly(double _X, const double * _Tab, int _N); 
#line 209
long double __cdecl _ldpoly(long double _X, const long double * _Tab, int _N); 
#line 210
float __cdecl _fdpoly(float _X, const float * _Tab, int _N); 
#line 212
double __cdecl _dlog(double _X, int _Baseflag); 
#line 213
long double __cdecl _ldlog(long double _X, int _Baseflag); 
#line 214
float __cdecl _fdlog(float _X, int _Baseflag); 
#line 216
double __cdecl _dsin(double _X, unsigned _Qoff); 
#line 217
long double __cdecl _ldsin(long double _X, unsigned _Qoff); 
#line 218
float __cdecl _fdsin(float _X, unsigned _Qoff); 
#line 225
typedef 
#line 222
union { 
#line 223
unsigned short _Sh[4]; 
#line 224
double _Val; 
#line 225
} _double_val; 
#line 232
typedef 
#line 229
union { 
#line 230
unsigned short _Sh[2]; 
#line 231
float _Val; 
#line 232
} _float_val; 
#line 239
typedef 
#line 236
union { 
#line 237
unsigned short _Sh[4]; 
#line 238
long double _Val; 
#line 239
} _ldouble_val; 
#line 247
typedef 
#line 242
union { 
#line 243
unsigned short _Word[4]; 
#line 244
float _Float; 
#line 245
double _Double; 
#line 246
long double _Long_double; 
#line 247
} _float_const; 
#line 249
extern const _float_const _Denorm_C, _Inf_C, _Nan_C, _Snan_C, _Hugeval_C; 
#line 250
extern const _float_const _FDenorm_C, _FInf_C, _FNan_C, _FSnan_C; 
#line 251
extern const _float_const _LDenorm_C, _LInf_C, _LNan_C, _LSnan_C; 
#line 253
extern const _float_const _Eps_C, _Rteps_C; 
#line 254
extern const _float_const _FEps_C, _FRteps_C; 
#line 255
extern const _float_const _LEps_C, _LRteps_C; 
#line 257
extern const double _Zero_C, _Xbig_C; 
#line 258
extern const float _FZero_C, _FXbig_C; 
#line 259
extern const long double _LZero_C, _LXbig_C; 
#line 288
extern "C++" {
#line 290
inline int fpclassify(float _X) throw() 
#line 291
{ 
#line 292
return _fdtest(&_X); 
#line 293
} 
#line 295
inline int fpclassify(double _X) throw() 
#line 296
{ 
#line 297
return _dtest(&_X); 
#line 298
} 
#line 300
inline int fpclassify(long double _X) throw() 
#line 301
{ 
#line 302
return _ldtest(&_X); 
#line 303
} 
#line 305
inline bool signbit(float _X) throw() 
#line 306
{ 
#line 307
return _fdsign(_X) != 0; 
#line 308
} 
#line 310
inline bool signbit(double _X) throw() 
#line 311
{ 
#line 312
return _dsign(_X) != 0; 
#line 313
} 
#line 315
inline bool signbit(long double _X) throw() 
#line 316
{ 
#line 317
return _ldsign(_X) != 0; 
#line 318
} 
#line 320
inline int _fpcomp(float _X, float _Y) throw() 
#line 321
{ 
#line 322
return _fdpcomp(_X, _Y); 
#line 323
} 
#line 325
inline int _fpcomp(double _X, double _Y) throw() 
#line 326
{ 
#line 327
return _dpcomp(_X, _Y); 
#line 328
} 
#line 330
inline int _fpcomp(long double _X, long double _Y) throw() 
#line 331
{ 
#line 332
return _ldpcomp(_X, _Y); 
#line 333
} 
#line 335
template< class _Trc, class _Tre> struct _Combined_type { 
#line 337
typedef float _Type; 
#line 338
}; 
#line 340
template<> struct _Combined_type< float, double>  { 
#line 342
typedef double _Type; 
#line 343
}; 
#line 345
template<> struct _Combined_type< float, long double>  { 
#line 347
typedef long double _Type; 
#line 348
}; 
#line 350
template< class _Ty, class _T2> struct _Real_widened { 
#line 352
typedef long double _Type; 
#line 353
}; 
#line 355
template<> struct _Real_widened< float, float>  { 
#line 357
typedef float _Type; 
#line 358
}; 
#line 360
template<> struct _Real_widened< float, double>  { 
#line 362
typedef double _Type; 
#line 363
}; 
#line 365
template<> struct _Real_widened< double, float>  { 
#line 367
typedef double _Type; 
#line 368
}; 
#line 370
template<> struct _Real_widened< double, double>  { 
#line 372
typedef double _Type; 
#line 373
}; 
#line 375
template< class _Ty> struct _Real_type { 
#line 377
typedef double _Type; 
#line 378
}; 
#line 380
template<> struct _Real_type< float>  { 
#line 382
typedef float _Type; 
#line 383
}; 
#line 385
template<> struct _Real_type< long double>  { 
#line 387
typedef long double _Type; 
#line 388
}; 
#line 390
template < class _T1, class _T2 >
      inline int _fpcomp ( _T1 _X, _T2 _Y ) throw ( )
    {
        typedef typename _Combined_type < float,
            typename _Real_widened <
            typename _Real_type < _T1 > :: _Type,
            typename _Real_type < _T2 > :: _Type > :: _Type > :: _Type _Tw;
        return _fpcomp ( ( _Tw ) _X, ( _Tw ) _Y );
    }
#line 400
template < class _Ty >
      inline bool isfinite ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) <= 0;
    }
#line 406
template < class _Ty >
      inline bool isinf ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 1;
    }
#line 412
template < class _Ty >
      inline bool isnan ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 2;
    }
#line 418
template < class _Ty >
      inline bool isnormal ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == ( - 1 );
    }
#line 424
template < class _Ty1, class _Ty2 >
      inline bool isgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 4 ) != 0;
    }
#line 430
template < class _Ty1, class _Ty2 >
      inline bool isgreaterequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 2 | 4 ) ) != 0;
    }
#line 436
template < class _Ty1, class _Ty2 >
      inline bool isless ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 1 ) != 0;
    }
#line 442
template < class _Ty1, class _Ty2 >
      inline bool islessequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 2 ) ) != 0;
    }
#line 448
template < class _Ty1, class _Ty2 >
      inline bool islessgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 4 ) ) != 0;
    }
#line 454
template < class _Ty1, class _Ty2 >
      inline bool isunordered ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return _fpcomp ( _X, _Y ) == 0;
    }
#line 459
}
#line 466 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
int __cdecl abs(int _X); 
#line 467
long __cdecl labs(long _X); 
#line 468
__int64 __cdecl llabs(__int64 _X); 
#line 470
double __cdecl acos(double _X); 
#line 471
double __cdecl asin(double _X); 
#line 472
double __cdecl atan(double _X); 
#line 473
double __cdecl atan2(double _Y, double _X); 
#line 475
double __cdecl cos(double _X); 
#line 476
double __cdecl cosh(double _X); 
#line 477
double __cdecl exp(double _X); 
#line 478
double __cdecl fabs(double _X); 
#line 479
double __cdecl fmod(double _X, double _Y); 
#line 480
double __cdecl log(double _X); 
#line 481
double __cdecl log10(double _X); 
#line 482
double __cdecl pow(double _X, double _Y); 
#line 483
double __cdecl sin(double _X); 
#line 484
double __cdecl sinh(double _X); 
#line 485
double __cdecl sqrt(double _X); 
#line 486
double __cdecl tan(double _X); 
#line 487
double __cdecl tanh(double _X); 
#line 489
double __cdecl acosh(double _X); 
#line 490
double __cdecl asinh(double _X); 
#line 491
double __cdecl atanh(double _X); 
#line 492
double __cdecl atof(const char * _String); 
#line 493
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 494
double __cdecl _cabs(_complex _Complex_value); 
#line 495
double __cdecl cbrt(double _X); 
#line 496
double __cdecl ceil(double _X); 
#line 497
double __cdecl _chgsign(double _X); 
#line 498
double __cdecl copysign(double _Number, double _Sign); 
#line 499
double __cdecl _copysign(double _Number, double _Sign); 
#line 500
double __cdecl erf(double _X); 
#line 501
double __cdecl erfc(double _X); 
#line 502
double __cdecl exp2(double _X); 
#line 503
double __cdecl expm1(double _X); 
#line 504
double __cdecl fdim(double _X, double _Y); 
#line 505
double __cdecl floor(double _X); 
#line 506
double __cdecl fma(double _X, double _Y, double _Z); 
#line 507
double __cdecl fmax(double _X, double _Y); 
#line 508
double __cdecl fmin(double _X, double _Y); 
#line 509
double __cdecl frexp(double _X, int * _Y); 
#line 510
double __cdecl hypot(double _X, double _Y); 
#line 511
double __cdecl _hypot(double _X, double _Y); 
#line 512
int __cdecl ilogb(double _X); 
#line 513
double __cdecl ldexp(double _X, int _Y); 
#line 514
double __cdecl lgamma(double _X); 
#line 515
__int64 __cdecl llrint(double _X); 
#line 516
__int64 __cdecl llround(double _X); 
#line 517
double __cdecl log1p(double _X); 
#line 518
double __cdecl log2(double _X); 
#line 519
double __cdecl logb(double _X); 
#line 520
long __cdecl lrint(double _X); 
#line 521
long __cdecl lround(double _X); 
#line 523
int __cdecl _matherr(_exception * _Except); 
#line 525
double __cdecl modf(double _X, double * _Y); 
#line 526
double __cdecl nan(const char * _X); 
#line 527
double __cdecl nearbyint(double _X); 
#line 528
double __cdecl nextafter(double _X, double _Y); 
#line 529
double __cdecl nexttoward(double _X, long double _Y); 
#line 530
double __cdecl remainder(double _X, double _Y); 
#line 531
double __cdecl remquo(double _X, double _Y, int * _Z); 
#line 532
double __cdecl rint(double _X); 
#line 533
double __cdecl round(double _X); 
#line 534
double __cdecl scalbln(double _X, long _Y); 
#line 535
double __cdecl scalbn(double _X, int _Y); 
#line 536
double __cdecl tgamma(double _X); 
#line 537
double __cdecl trunc(double _X); 
#line 538
double __cdecl _j0(double _X); 
#line 539
double __cdecl _j1(double _X); 
#line 540
double __cdecl _jn(int _X, double _Y); 
#line 541
double __cdecl _y0(double _X); 
#line 542
double __cdecl _y1(double _X); 
#line 543
double __cdecl _yn(int _X, double _Y); 
#line 545
float __cdecl acoshf(float _X); 
#line 546
float __cdecl asinhf(float _X); 
#line 547
float __cdecl atanhf(float _X); 
#line 548
float __cdecl cbrtf(float _X); 
#line 549
float __cdecl _chgsignf(float _X); 
#line 550
float __cdecl copysignf(float _Number, float _Sign); 
#line 551
float __cdecl _copysignf(float _Number, float _Sign); 
#line 552
float __cdecl erff(float _X); 
#line 553
float __cdecl erfcf(float _X); 
#line 554
float __cdecl expm1f(float _X); 
#line 555
float __cdecl exp2f(float _X); 
#line 556
float __cdecl fdimf(float _X, float _Y); 
#line 557
float __cdecl fmaf(float _X, float _Y, float _Z); 
#line 558
float __cdecl fmaxf(float _X, float _Y); 
#line 559
float __cdecl fminf(float _X, float _Y); 
#line 560
float __cdecl _hypotf(float _X, float _Y); 
#line 561
int __cdecl ilogbf(float _X); 
#line 562
float __cdecl lgammaf(float _X); 
#line 563
__int64 __cdecl llrintf(float _X); 
#line 564
__int64 __cdecl llroundf(float _X); 
#line 565
float __cdecl log1pf(float _X); 
#line 566
float __cdecl log2f(float _X); 
#line 567
float __cdecl logbf(float _X); 
#line 568
long __cdecl lrintf(float _X); 
#line 569
long __cdecl lroundf(float _X); 
#line 570
float __cdecl nanf(const char * _X); 
#line 571
float __cdecl nearbyintf(float _X); 
#line 572
float __cdecl nextafterf(float _X, float _Y); 
#line 573
float __cdecl nexttowardf(float _X, long double _Y); 
#line 574
float __cdecl remainderf(float _X, float _Y); 
#line 575
float __cdecl remquof(float _X, float _Y, int * _Z); 
#line 576
float __cdecl rintf(float _X); 
#line 577
float __cdecl roundf(float _X); 
#line 578
float __cdecl scalblnf(float _X, long _Y); 
#line 579
float __cdecl scalbnf(float _X, int _Y); 
#line 580
float __cdecl tgammaf(float _X); 
#line 581
float __cdecl truncf(float _X); 
#line 591 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
float __cdecl _logbf(float _X); 
#line 592
float __cdecl _nextafterf(float _X, float _Y); 
#line 593
int __cdecl _finitef(float _X); 
#line 594
int __cdecl _isnanf(float _X); 
#line 595
int __cdecl _fpclassf(float _X); 
#line 597
int __cdecl _set_FMA3_enable(int _Flag); 
#line 598
int __cdecl _get_FMA3_enable(); 
#line 611 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
float __cdecl acosf(float _X); 
#line 612
float __cdecl asinf(float _X); 
#line 613
float __cdecl atan2f(float _Y, float _X); 
#line 614
float __cdecl atanf(float _X); 
#line 615
float __cdecl ceilf(float _X); 
#line 616
float __cdecl cosf(float _X); 
#line 617
float __cdecl coshf(float _X); 
#line 618
float __cdecl expf(float _X); 
#line 670 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
__inline float __cdecl fabsf(float _X) 
#line 671
{ 
#line 672
return (float)fabs(_X); 
#line 673
} 
#line 679 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
float __cdecl floorf(float _X); 
#line 680
float __cdecl fmodf(float _X, float _Y); 
#line 696 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
__inline float __cdecl frexpf(float _X, int *_Y) 
#line 697
{ 
#line 698
return (float)frexp(_X, _Y); 
#line 699
} 
#line 701
__inline float __cdecl hypotf(float _X, float _Y) 
#line 702
{ 
#line 703
return _hypotf(_X, _Y); 
#line 704
} 
#line 706
__inline float __cdecl ldexpf(float _X, int _Y) 
#line 707
{ 
#line 708
return (float)ldexp(_X, _Y); 
#line 709
} 
#line 713
float __cdecl log10f(float _X); 
#line 714
float __cdecl logf(float _X); 
#line 715
float __cdecl modff(float _X, float * _Y); 
#line 716
float __cdecl powf(float _X, float _Y); 
#line 717
float __cdecl sinf(float _X); 
#line 718
float __cdecl sinhf(float _X); 
#line 719
float __cdecl sqrtf(float _X); 
#line 720
float __cdecl tanf(float _X); 
#line 721
float __cdecl tanhf(float _X); 
#line 775 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
long double __cdecl acoshl(long double _X); 
#line 777
__inline long double __cdecl acosl(long double _X) 
#line 778
{ 
#line 779
return acos((double)_X); 
#line 780
} 
#line 782
long double __cdecl asinhl(long double _X); 
#line 784
__inline long double __cdecl asinl(long double _X) 
#line 785
{ 
#line 786
return asin((double)_X); 
#line 787
} 
#line 789
__inline long double __cdecl atan2l(long double _Y, long double _X) 
#line 790
{ 
#line 791
return atan2((double)_Y, (double)_X); 
#line 792
} 
#line 794
long double __cdecl atanhl(long double _X); 
#line 796
__inline long double __cdecl atanl(long double _X) 
#line 797
{ 
#line 798
return atan((double)_X); 
#line 799
} 
#line 801
long double __cdecl cbrtl(long double _X); 
#line 803
__inline long double __cdecl ceill(long double _X) 
#line 804
{ 
#line 805
return ceil((double)_X); 
#line 806
} 
#line 808
__inline long double __cdecl _chgsignl(long double _X) 
#line 809
{ 
#line 810
return _chgsign((double)_X); 
#line 811
} 
#line 813
long double __cdecl copysignl(long double _Number, long double _Sign); 
#line 815
__inline long double __cdecl _copysignl(long double _Number, long double _Sign) 
#line 816
{ 
#line 817
return _copysign((double)_Number, (double)_Sign); 
#line 818
} 
#line 820
__inline long double __cdecl coshl(long double _X) 
#line 821
{ 
#line 822
return cosh((double)_X); 
#line 823
} 
#line 825
__inline long double __cdecl cosl(long double _X) 
#line 826
{ 
#line 827
return cos((double)_X); 
#line 828
} 
#line 830
long double __cdecl erfl(long double _X); 
#line 831
long double __cdecl erfcl(long double _X); 
#line 833
__inline long double __cdecl expl(long double _X) 
#line 834
{ 
#line 835
return exp((double)_X); 
#line 836
} 
#line 838
long double __cdecl exp2l(long double _X); 
#line 839
long double __cdecl expm1l(long double _X); 
#line 841
__inline long double __cdecl fabsl(long double _X) 
#line 842
{ 
#line 843
return fabs((double)_X); 
#line 844
} 
#line 846
long double __cdecl fdiml(long double _X, long double _Y); 
#line 848
__inline long double __cdecl floorl(long double _X) 
#line 849
{ 
#line 850
return floor((double)_X); 
#line 851
} 
#line 853
long double __cdecl fmal(long double _X, long double _Y, long double _Z); 
#line 854
long double __cdecl fmaxl(long double _X, long double _Y); 
#line 855
long double __cdecl fminl(long double _X, long double _Y); 
#line 857
__inline long double __cdecl fmodl(long double _X, long double _Y) 
#line 858
{ 
#line 859
return fmod((double)_X, (double)_Y); 
#line 860
} 
#line 862
__inline long double __cdecl frexpl(long double _X, int *_Y) 
#line 863
{ 
#line 864
return frexp((double)_X, _Y); 
#line 865
} 
#line 867
int __cdecl ilogbl(long double _X); 
#line 869
__inline long double __cdecl _hypotl(long double _X, long double _Y) 
#line 870
{ 
#line 871
return _hypot((double)_X, (double)_Y); 
#line 872
} 
#line 874
__inline long double __cdecl hypotl(long double _X, long double _Y) 
#line 875
{ 
#line 876
return _hypot((double)_X, (double)_Y); 
#line 877
} 
#line 879
__inline long double __cdecl ldexpl(long double _X, int _Y) 
#line 880
{ 
#line 881
return ldexp((double)_X, _Y); 
#line 882
} 
#line 884
long double __cdecl lgammal(long double _X); 
#line 885
__int64 __cdecl llrintl(long double _X); 
#line 886
__int64 __cdecl llroundl(long double _X); 
#line 888
__inline long double __cdecl logl(long double _X) 
#line 889
{ 
#line 890
return log((double)_X); 
#line 891
} 
#line 893
__inline long double __cdecl log10l(long double _X) 
#line 894
{ 
#line 895
return log10((double)_X); 
#line 896
} 
#line 898
long double __cdecl log1pl(long double _X); 
#line 899
long double __cdecl log2l(long double _X); 
#line 900
long double __cdecl logbl(long double _X); 
#line 901
long __cdecl lrintl(long double _X); 
#line 902
long __cdecl lroundl(long double _X); 
#line 904
__inline long double __cdecl modfl(long double _X, long double *_Y) 
#line 905
{ 
#line 906
double _F, _I; 
#line 907
_F = modf((double)_X, &_I); 
#line 908
(*_Y) = _I; 
#line 909
return _F; 
#line 910
} 
#line 912
long double __cdecl nanl(const char * _X); 
#line 913
long double __cdecl nearbyintl(long double _X); 
#line 914
long double __cdecl nextafterl(long double _X, long double _Y); 
#line 915
long double __cdecl nexttowardl(long double _X, long double _Y); 
#line 917
__inline long double __cdecl powl(long double _X, long double _Y) 
#line 918
{ 
#line 919
return pow((double)_X, (double)_Y); 
#line 920
} 
#line 922
long double __cdecl remainderl(long double _X, long double _Y); 
#line 923
long double __cdecl remquol(long double _X, long double _Y, int * _Z); 
#line 924
long double __cdecl rintl(long double _X); 
#line 925
long double __cdecl roundl(long double _X); 
#line 926
long double __cdecl scalblnl(long double _X, long _Y); 
#line 927
long double __cdecl scalbnl(long double _X, int _Y); 
#line 929
__inline long double __cdecl sinhl(long double _X) 
#line 930
{ 
#line 931
return sinh((double)_X); 
#line 932
} 
#line 934
__inline long double __cdecl sinl(long double _X) 
#line 935
{ 
#line 936
return sin((double)_X); 
#line 937
} 
#line 939
__inline long double __cdecl sqrtl(long double _X) 
#line 940
{ 
#line 941
return sqrt((double)_X); 
#line 942
} 
#line 944
__inline long double __cdecl tanhl(long double _X) 
#line 945
{ 
#line 946
return tanh((double)_X); 
#line 947
} 
#line 949
__inline long double __cdecl tanl(long double _X) 
#line 950
{ 
#line 951
return tan((double)_X); 
#line 952
} 
#line 954
long double __cdecl tgammal(long double _X); 
#line 955
long double __cdecl truncl(long double _X); 
#line 976 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
extern double HUGE; 
#line 981 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
double __cdecl j0(double _X); 
#line 982
double __cdecl j1(double _X); 
#line 983
double __cdecl jn(int _X, double _Y); 
#line 984
double __cdecl y0(double _X); 
#line 985
double __cdecl y1(double _X); 
#line 986
double __cdecl yn(int _X, double _Y); 
#line 994 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_math.h"
}
#line 992
#pragma warning(pop)
#line 994
__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_malloc.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 54 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_malloc.h"
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 55
_calloc_base(size_t _Count, size_t _Size); 
#line 61
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 62
calloc(size_t _Count, size_t _Size); 
#line 68
int __cdecl _callnewh(size_t _Size); 
#line 73
__declspec(allocator) void *__cdecl 
#line 74
_expand(void * _Block, size_t _Size); 
#line 80
void __cdecl _free_base(void * _Block); 
#line 85
void __cdecl free(void * _Block); 
#line 90
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 91
_malloc_base(size_t _Size); 
#line 96
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 97
malloc(size_t _Size); 
#line 103
size_t __cdecl _msize_base(void * _Block); 
#line 109
size_t __cdecl _msize(void * _Block); 
#line 114
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 115
_realloc_base(void * _Block, size_t _Size); 
#line 121
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 122
realloc(void * _Block, size_t _Size); 
#line 128
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 129
_recalloc_base(void * _Block, size_t _Count, size_t _Size); 
#line 136
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 137
_recalloc(void * _Block, size_t _Count, size_t _Size); 
#line 144
void __cdecl _aligned_free(void * _Block); 
#line 149
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 150
_aligned_malloc(size_t _Size, size_t _Alignment); 
#line 156
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 157
_aligned_offset_malloc(size_t _Size, size_t _Alignment, size_t _Offset); 
#line 165
size_t __cdecl _aligned_msize(void * _Block, size_t _Alignment, size_t _Offset); 
#line 172
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 173
_aligned_offset_realloc(void * _Block, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 181
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 182
_aligned_offset_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 191
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 192
_aligned_realloc(void * _Block, size_t _Size, size_t _Alignment); 
#line 199
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 200
_aligned_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment); 
#line 228 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_malloc.h"
}__pragma( pack ( pop )) 
#line 16 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_search.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
typedef int (__cdecl *_CoreCrtSecureSearchSortCompareFunction)(void *, const void *, const void *); 
#line 20
typedef int (__cdecl *_CoreCrtNonSecureSearchSortCompareFunction)(const void *, const void *); 
#line 26
void *__cdecl bsearch_s(const void * _Key, const void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 35
void __cdecl qsort_s(void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 48 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_search.h"
void *__cdecl bsearch(const void * _Key, const void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 56
void __cdecl qsort(void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 64
void *__cdecl _lfind_s(const void * _Key, const void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 74
void *__cdecl _lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 83
void *__cdecl _lsearch_s(const void * _Key, void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 93
void *__cdecl _lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 191 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_search.h"
void *__cdecl lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 200
void *__cdecl lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 212 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_search.h"
}__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 50
errno_t __cdecl _itow_s(int _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 57
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itow_s ( int _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 64 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _itow(int _Value, __wchar_t * _Buffer, int _Radix); 
#line 73 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ltow_s(long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 80
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltow_s ( long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 87 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ltow(long _Value, __wchar_t * _Buffer, int _Radix); 
#line 95 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ultow_s(unsigned long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 102
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultow_s ( unsigned long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 109 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ultow(unsigned long _Value, __wchar_t * _Buffer, int _Radix); 
#line 117 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
double __cdecl wcstod(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 123
double __cdecl _wcstod_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 130
long __cdecl wcstol(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 137
long __cdecl _wcstol_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 145
__int64 __cdecl wcstoll(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 152
__int64 __cdecl _wcstoll_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 160
unsigned long __cdecl wcstoul(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 167
unsigned long __cdecl _wcstoul_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 175
unsigned __int64 __cdecl wcstoull(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 182
unsigned __int64 __cdecl _wcstoull_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 190
long double __cdecl wcstold(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 196
long double __cdecl _wcstold_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 203
float __cdecl wcstof(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 209
float __cdecl _wcstof_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 216
double __cdecl _wtof(const __wchar_t * _String); 
#line 221
double __cdecl _wtof_l(const __wchar_t * _String, _locale_t _Locale); 
#line 227
int __cdecl _wtoi(const __wchar_t * _String); 
#line 232
int __cdecl _wtoi_l(const __wchar_t * _String, _locale_t _Locale); 
#line 238
long __cdecl _wtol(const __wchar_t * _String); 
#line 243
long __cdecl _wtol_l(const __wchar_t * _String, _locale_t _Locale); 
#line 249
__int64 __cdecl _wtoll(const __wchar_t * _String); 
#line 254
__int64 __cdecl _wtoll_l(const __wchar_t * _String, _locale_t _Locale); 
#line 260
errno_t __cdecl _i64tow_s(__int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 268
__wchar_t *__cdecl _i64tow(__int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 275
errno_t __cdecl _ui64tow_s(unsigned __int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 283
__wchar_t *__cdecl _ui64tow(unsigned __int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 290
__int64 __cdecl _wtoi64(const __wchar_t * _String); 
#line 295
__int64 __cdecl _wtoi64_l(const __wchar_t * _String, _locale_t _Locale); 
#line 301
__int64 __cdecl _wcstoi64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 308
__int64 __cdecl _wcstoi64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 316
unsigned __int64 __cdecl _wcstoui64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 323
unsigned __int64 __cdecl _wcstoui64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 335
__declspec(allocator) __wchar_t *__cdecl _wfullpath(__wchar_t * _Buffer, const __wchar_t * _Path, size_t _BufferCount); 
#line 344
errno_t __cdecl _wmakepath_s(__wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 353
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wmakepath_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _Drive, wchar_t const * _Dir, wchar_t const * _Filename, wchar_t const * _Ext ) throw ( ) { return _wmakepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 362 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
void __cdecl _wmakepath(__wchar_t * _Buffer, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 373 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
#pragma warning(pop)
#line 375
void __cdecl _wperror(const __wchar_t * _ErrorMessage); 
#line 380
void __cdecl _wsplitpath(const __wchar_t * _FullPath, __wchar_t * _Drive, __wchar_t * _Dir, __wchar_t * _Filename, __wchar_t * _Ext); 
#line 388
errno_t __cdecl _wsplitpath_s(const __wchar_t * _FullPath, __wchar_t * _Drive, size_t _DriveCount, __wchar_t * _Dir, size_t _DirCount, __wchar_t * _Filename, size_t _FilenameCount, __wchar_t * _Ext, size_t _ExtCount); 
#line 400
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _wsplitpath_s ( wchar_t const * _Path, wchar_t ( & _Drive ) [ _DriveSize ], wchar_t ( & _Dir ) [ _DirSize ], wchar_t ( & _Name ) [ _NameSize ], wchar_t ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _wsplitpath_s ( _Path, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 409 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _wdupenv_s(__wchar_t ** _Buffer, size_t * _BufferCount, const __wchar_t * _VarName); 
#line 418
__wchar_t *__cdecl _wgetenv(const __wchar_t * _VarName); 
#line 424
errno_t __cdecl _wgetenv_s(size_t * _RequiredCount, __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _VarName); 
#line 431
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wgetenv_s ( size_t * _RequiredCount, wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _VarName ) throw ( ) { return _wgetenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 440 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wputenv(const __wchar_t * _EnvString); 
#line 445
errno_t __cdecl _wputenv_s(const __wchar_t * _Name, const __wchar_t * _Value); 
#line 450
errno_t __cdecl _wsearchenv_s(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _Buffer, size_t _BufferCount); 
#line 457
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wsearchenv_s ( wchar_t const * _Filename, wchar_t const * _VarName, wchar_t ( & _ResultPath ) [ _Size ] ) throw ( ) { return _wsearchenv_s ( _Filename, _VarName, _ResultPath, _Size ); }}
#line 464 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wsearchenv(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _ResultPath); 
#line 471 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wsystem(const __wchar_t * _Command); 
#line 479 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\corecrt_wstdlib.h"
}__pragma( pack ( pop )) 
#line 18 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 34 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
void __cdecl _swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 52
__declspec(noreturn) void __cdecl exit(int _Code); 
#line 53
__declspec(noreturn) void __cdecl _exit(int _Code); 
#line 54
__declspec(noreturn) void __cdecl _Exit(int _Code); 
#line 55
__declspec(noreturn) void __cdecl quick_exit(int _Code); 
#line 56
__declspec(noreturn) void __cdecl abort(); 
#line 63 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
unsigned __cdecl _set_abort_behavior(unsigned _Flags, unsigned _Mask); 
#line 73
typedef int (__cdecl *_onexit_t)(void); 
#line 140 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
int __cdecl atexit(void (__cdecl *)(void)); 
#line 141
_onexit_t __cdecl _onexit(_onexit_t _Func); 
#line 144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
int __cdecl at_quick_exit(void (__cdecl *)(void)); 
#line 155
typedef void (__cdecl *_purecall_handler)(void); 
#line 158
typedef void (__cdecl *_invalid_parameter_handler)(const __wchar_t *, const __wchar_t *, const __wchar_t *, unsigned, uintptr_t); 
#line 167
_purecall_handler __cdecl _set_purecall_handler(_purecall_handler _Handler); 
#line 171
_purecall_handler __cdecl _get_purecall_handler(); 
#line 174
_invalid_parameter_handler __cdecl _set_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 178
_invalid_parameter_handler __cdecl _get_invalid_parameter_handler(); 
#line 180
_invalid_parameter_handler __cdecl _set_thread_local_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 184
_invalid_parameter_handler __cdecl _get_thread_local_invalid_parameter_handler(); 
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
int __cdecl _set_error_mode(int _Mode); 
#line 213
int *__cdecl _errno(); 
#line 216
errno_t __cdecl _set_errno(int _Value); 
#line 217
errno_t __cdecl _get_errno(int * _Value); 
#line 219
unsigned long *__cdecl __doserrno(); 
#line 222
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 223
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 226
char **__cdecl __sys_errlist(); 
#line 229
int *__cdecl __sys_nerr(); 
#line 232
void __cdecl perror(const char * _ErrMsg); 
#line 238 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
char **__cdecl __p__pgmptr(); 
#line 239
__wchar_t **__cdecl __p__wpgmptr(); 
#line 240
int *__cdecl __p__fmode(); 
#line 255 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
errno_t __cdecl _get_pgmptr(char ** _Value); 
#line 258
errno_t __cdecl _get_wpgmptr(__wchar_t ** _Value); 
#line 260
errno_t __cdecl _set_fmode(int _Mode); 
#line 262
errno_t __cdecl _get_fmode(int * _PMode); 
#line 275
typedef 
#line 271
struct _div_t { 
#line 273
int quot; 
#line 274
int rem; 
#line 275
} div_t; 
#line 281
typedef 
#line 277
struct _ldiv_t { 
#line 279
long quot; 
#line 280
long rem; 
#line 281
} ldiv_t; 
#line 287
typedef 
#line 283
struct _lldiv_t { 
#line 285
__int64 quot; 
#line 286
__int64 rem; 
#line 287
} lldiv_t; 
#line 289
int __cdecl abs(int _Number); 
#line 290
long __cdecl labs(long _Number); 
#line 291
__int64 __cdecl llabs(__int64 _Number); 
#line 292
__int64 __cdecl _abs64(__int64 _Number); 
#line 294
unsigned short __cdecl _byteswap_ushort(unsigned short _Number); 
#line 295
unsigned long __cdecl _byteswap_ulong(unsigned long _Number); 
#line 296
unsigned __int64 __cdecl _byteswap_uint64(unsigned __int64 _Number); 
#line 298
div_t __cdecl div(int _Numerator, int _Denominator); 
#line 299
ldiv_t __cdecl ldiv(long _Numerator, long _Denominator); 
#line 300
lldiv_t __cdecl lldiv(__int64 _Numerator, __int64 _Denominator); 
#line 304
#pragma warning (push)
#pragma warning (disable:6540)
#line 307
unsigned __cdecl _rotl(unsigned _Value, int _Shift); 
#line 313
unsigned long __cdecl _lrotl(unsigned long _Value, int _Shift); 
#line 318
unsigned __int64 __cdecl _rotl64(unsigned __int64 _Value, int _Shift); 
#line 323
unsigned __cdecl _rotr(unsigned _Value, int _Shift); 
#line 329
unsigned long __cdecl _lrotr(unsigned long _Value, int _Shift); 
#line 334
unsigned __int64 __cdecl _rotr64(unsigned __int64 _Value, int _Shift); 
#line 339
#pragma warning (pop)
#line 346
void __cdecl srand(unsigned _Seed); 
#line 348
int __cdecl rand(); 
#line 357 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
extern "C++" {
#line 359
inline long abs(const long _X) throw() 
#line 360
{ 
#line 361
return labs(_X); 
#line 362
} 
#line 364
inline __int64 abs(const __int64 _X) throw() 
#line 365
{ 
#line 366
return llabs(_X); 
#line 367
} 
#line 369
inline ldiv_t div(const long _A1, const long _A2) throw() 
#line 370
{ 
#line 371
return ldiv(_A1, _A2); 
#line 372
} 
#line 374
inline lldiv_t div(const __int64 _A1, const __int64 _A2) throw() 
#line 375
{ 
#line 376
return lldiv(_A1, _A2); 
#line 377
} 
#line 378
}
#line 390 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
#pragma pack ( push, 4 )
#line 394
typedef 
#line 392
struct { 
#line 393
unsigned char ld[10]; 
#line 394
} _LDOUBLE; 
#pragma pack ( pop )
#line 414 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
typedef 
#line 412
struct { 
#line 413
double x; 
#line 414
} _CRT_DOUBLE; 
#line 419
typedef 
#line 417
struct { 
#line 418
float f; 
#line 419
} _CRT_FLOAT; 
#line 428
typedef 
#line 426
struct { 
#line 427
long double x; 
#line 428
} _LONGDOUBLE; 
#line 432
#pragma pack ( push, 4 )
#line 436
typedef 
#line 434
struct { 
#line 435
unsigned char ld12[12]; 
#line 436
} _LDBL12; 
#pragma pack ( pop )
#line 446
double __cdecl atof(const char * _String); 
#line 447
int __cdecl atoi(const char * _String); 
#line 448
long __cdecl atol(const char * _String); 
#line 449
__int64 __cdecl atoll(const char * _String); 
#line 450
__int64 __cdecl _atoi64(const char * _String); 
#line 452
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 453
int __cdecl _atoi_l(const char * _String, _locale_t _Locale); 
#line 454
long __cdecl _atol_l(const char * _String, _locale_t _Locale); 
#line 455
__int64 __cdecl _atoll_l(const char * _String, _locale_t _Locale); 
#line 456
__int64 __cdecl _atoi64_l(const char * _String, _locale_t _Locale); 
#line 458
int __cdecl _atoflt(_CRT_FLOAT * _Result, const char * _String); 
#line 459
int __cdecl _atodbl(_CRT_DOUBLE * _Result, char * _String); 
#line 460
int __cdecl _atoldbl(_LDOUBLE * _Result, char * _String); 
#line 463
int __cdecl _atoflt_l(_CRT_FLOAT * _Result, const char * _String, _locale_t _Locale); 
#line 470
int __cdecl _atodbl_l(_CRT_DOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 478
int __cdecl _atoldbl_l(_LDOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 485
float __cdecl strtof(const char * _String, char ** _EndPtr); 
#line 491
float __cdecl _strtof_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 498
double __cdecl strtod(const char * _String, char ** _EndPtr); 
#line 504
double __cdecl _strtod_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 511
long double __cdecl strtold(const char * _String, char ** _EndPtr); 
#line 517
long double __cdecl _strtold_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 524
long __cdecl strtol(const char * _String, char ** _EndPtr, int _Radix); 
#line 531
long __cdecl _strtol_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 539
__int64 __cdecl strtoll(const char * _String, char ** _EndPtr, int _Radix); 
#line 546
__int64 __cdecl _strtoll_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 554
unsigned long __cdecl strtoul(const char * _String, char ** _EndPtr, int _Radix); 
#line 561
unsigned long __cdecl _strtoul_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 569
unsigned __int64 __cdecl strtoull(const char * _String, char ** _EndPtr, int _Radix); 
#line 576
unsigned __int64 __cdecl _strtoull_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 584
__int64 __cdecl _strtoi64(const char * _String, char ** _EndPtr, int _Radix); 
#line 591
__int64 __cdecl _strtoi64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 599
unsigned __int64 __cdecl _strtoui64(const char * _String, char ** _EndPtr, int _Radix); 
#line 606
unsigned __int64 __cdecl _strtoui64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 622
errno_t __cdecl _itoa_s(int _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 629
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itoa_s ( int _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 637 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
char *__cdecl _itoa(int _Value, char * _Buffer, int _Radix); 
#line 646 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
#pragma warning(pop)
#line 650
errno_t __cdecl _ltoa_s(long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 657
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltoa_s ( long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 664 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
char *__cdecl _ltoa(long _Value, char * _Buffer, int _Radix); 
#line 673 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
errno_t __cdecl _ultoa_s(unsigned long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 680
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultoa_s ( unsigned long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 687 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 28726)
char *__cdecl _ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 695 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
#pragma warning(pop)
#line 699
errno_t __cdecl _i64toa_s(__int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 708
char *__cdecl _i64toa(__int64 _Value, char * _Buffer, int _Radix); 
#line 716
errno_t __cdecl _ui64toa_s(unsigned __int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 724
char *__cdecl _ui64toa(unsigned __int64 _Value, char * _Buffer, int _Radix); 
#line 744
errno_t __cdecl _ecvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 753
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ecvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _ecvt_s ( _Buffer, _Size, _Value, _DigitCount, _PtDec, _PtSign ); }}
#line 763 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
char *__cdecl _ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 772
errno_t __cdecl _fcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 781
extern "C++" {template < size_t _Size > inline errno_t __cdecl _fcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _fcvt_s ( _Buffer, _Size, _Value, _FractionalDigitCount, _PtDec, _PtSign ); }}
#line 793 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
char *__cdecl _fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 801
errno_t __cdecl _gcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount); 
#line 808
extern "C++" {template < size_t _Size > inline errno_t __cdecl _gcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount ) throw ( ) { return _gcvt_s ( _Buffer, _Size, _Value, _DigitCount ); }}
#line 817 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
char *__cdecl _gcvt(double _Value, int _DigitCount, char * _Buffer); 
#line 846 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
int __cdecl ___mb_cur_max_func(); 
#line 849
int __cdecl ___mb_cur_max_l_func(_locale_t _Locale); 
#line 855 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
int __cdecl mblen(const char * _Ch, size_t _MaxCount); 
#line 861
int __cdecl _mblen_l(const char * _Ch, size_t _MaxCount, _locale_t _Locale); 
#line 869
size_t __cdecl _mbstrlen(const char * _String); 
#line 875
size_t __cdecl _mbstrlen_l(const char * _String, _locale_t _Locale); 
#line 882
size_t __cdecl _mbstrnlen(const char * _String, size_t _MaxCount); 
#line 889
size_t __cdecl _mbstrnlen_l(const char * _String, size_t _MaxCount, _locale_t _Locale); 
#line 896
int __cdecl mbtowc(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes); 
#line 903
int __cdecl _mbtowc_l(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes, _locale_t _Locale); 
#line 911
errno_t __cdecl mbstowcs_s(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount); 
#line 919
extern "C++" {template < size_t _Size > inline errno_t __cdecl mbstowcs_s ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount ) throw ( ) { return mbstowcs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 927 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
size_t __cdecl mbstowcs(__wchar_t * _Dest, const char * _Source, size_t _MaxCount); 
#line 935 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
errno_t __cdecl _mbstowcs_s_l(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount, _locale_t _Locale); 
#line 944
extern "C++" {template < size_t _Size > inline errno_t __cdecl _mbstowcs_s_l ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _mbstowcs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 953 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
size_t __cdecl _mbstowcs_l(__wchar_t * _Dest, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 966 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
int __cdecl wctomb(char * _MbCh, __wchar_t _WCh); 
#line 972
int __cdecl _wctomb_l(char * _MbCh, __wchar_t _WCh, _locale_t _Locale); 
#line 981
errno_t __cdecl wctomb_s(int * _SizeConverted, char * _MbCh, rsize_t _SizeInBytes, __wchar_t _WCh); 
#line 991 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
errno_t __cdecl _wctomb_s_l(int * _SizeConverted, char * _MbCh, size_t _SizeInBytes, __wchar_t _WCh, _locale_t _Locale); 
#line 999
errno_t __cdecl wcstombs_s(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes); 
#line 1007
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcstombs_s ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount ) throw ( ) { return wcstombs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 1015 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
size_t __cdecl wcstombs(char * _Dest, const __wchar_t * _Source, size_t _MaxCount); 
#line 1023 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
errno_t __cdecl _wcstombs_s_l(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes, _locale_t _Locale); 
#line 1032
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcstombs_s_l ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _wcstombs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 1041 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
size_t __cdecl _wcstombs_l(char * _Dest, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 1071 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
__declspec(allocator) char *__cdecl _fullpath(char * _Buffer, const char * _Path, size_t _BufferCount); 
#line 1080
errno_t __cdecl _makepath_s(char * _Buffer, size_t _BufferCount, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1089
extern "C++" {template < size_t _Size > inline errno_t __cdecl _makepath_s ( char ( & _Buffer ) [ _Size ], char const * _Drive, char const * _Dir, char const * _Filename, char const * _Ext ) throw ( ) { return _makepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 1098 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
void __cdecl _makepath(char * _Buffer, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1109 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
#pragma warning(pop)
#line 1112
void __cdecl _splitpath(const char * _FullPath, char * _Drive, char * _Dir, char * _Filename, char * _Ext); 
#line 1121
errno_t __cdecl _splitpath_s(const char * _FullPath, char * _Drive, size_t _DriveCount, char * _Dir, size_t _DirCount, char * _Filename, size_t _FilenameCount, char * _Ext, size_t _ExtCount); 
#line 1133
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _splitpath_s ( char const * _Dest, char ( & _Drive ) [ _DriveSize ], char ( & _Dir ) [ _DirSize ], char ( & _Name ) [ _NameSize ], char ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _splitpath_s ( _Dest, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 1139
errno_t __cdecl getenv_s(size_t * _RequiredCount, char * _Buffer, rsize_t _BufferCount, const char * _VarName); 
#line 1151 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
int *__cdecl __p___argc(); 
#line 1152
char ***__cdecl __p___argv(); 
#line 1153
__wchar_t ***__cdecl __p___wargv(); 
#line 1165 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
char ***__cdecl __p__environ(); 
#line 1166
__wchar_t ***__cdecl __p__wenviron(); 
#line 1191 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
char *__cdecl getenv(const char * _VarName); 
#line 1195
extern "C++" {template < size_t _Size > inline errno_t __cdecl getenv_s ( size_t * _RequiredCount, char ( & _Buffer ) [ _Size ], char const * _VarName ) throw ( ) { return getenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 1208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
errno_t __cdecl _dupenv_s(char ** _Buffer, size_t * _BufferCount, const char * _VarName); 
#line 1218 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
int __cdecl system(const char * _Command); 
#line 1224
#pragma warning (push)
#pragma warning (disable:6540)
#line 1228
int __cdecl _putenv(const char * _EnvString); 
#line 1233
errno_t __cdecl _putenv_s(const char * _Name, const char * _Value); 
#line 1238
#pragma warning (pop)
#line 1240
errno_t __cdecl _searchenv_s(const char * _Filename, const char * _VarName, char * _Buffer, size_t _BufferCount); 
#line 1247
extern "C++" {template < size_t _Size > inline errno_t __cdecl _searchenv_s ( char const * _Filename, char const * _VarName, char ( & _Buffer ) [ _Size ] ) throw ( ) { return _searchenv_s ( _Filename, _VarName, _Buffer, _Size ); }}
#line 1254 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
void __cdecl _searchenv(const char * _Filename, const char * _VarName, char * _Buffer); 
#line 1262 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetErrorMode in" "stead. See online help for details.")) void __cdecl 
#line 1263
_seterrormode(int _Mode); 
#line 1267
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Beep instead. S" "ee online help for details.")) void __cdecl 
#line 1268
_beep(unsigned _Frequency, unsigned _Duration); 
#line 1273
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Sleep instead. " "See online help for details.")) void __cdecl 
#line 1274
_sleep(unsigned long _Duration); 
#line 1296 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 4141)
#line 1300
char *__cdecl ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 1308
char *__cdecl fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 1316
char *__cdecl gcvt(double _Value, int _DigitCount, char * _DstBuf); 
#line 1323
char *__cdecl itoa(int _Value, char * _Buffer, int _Radix); 
#line 1330
char *__cdecl ltoa(long _Value, char * _Buffer, int _Radix); 
#line 1338
void __cdecl swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 1345
char *__cdecl ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 1354
int __cdecl putenv(const char * _EnvString); 
#line 1358
#pragma warning(pop)
#line 1360
_onexit_t __cdecl onexit(_onexit_t _Func); 
#line 1366 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\stdlib.h"
}__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new.h"
extern "C++" {
#line 15
#pragma pack ( push, 8 )
#line 17
#pragma warning(push)
#pragma warning(disable: 4985)
#line 49 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new.h"
namespace std { 
#line 51
struct nothrow_t { 
#line 53
explicit nothrow_t() = default;
#line 55 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new.h"
}; 
#line 60
extern const nothrow_t nothrow; 
#line 62 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new.h"
}
#line 66 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new.h"
__declspec(allocator) void *__cdecl operator new(size_t _Size); 
#line 71
__declspec(allocator) void *__cdecl operator new(size_t _Size, const std::nothrow_t &) noexcept; 
#line 77
__declspec(allocator) void *__cdecl operator new[](size_t _Size); 
#line 82
__declspec(allocator) void *__cdecl operator new[](size_t _Size, const std::nothrow_t &) noexcept; 
#line 87
void __cdecl operator delete(void * _Block) noexcept; 
#line 91
void __cdecl operator delete(void * _Block, const std::nothrow_t &) noexcept; 
#line 96
void __cdecl operator delete[](void * _Block) noexcept; 
#line 100
void __cdecl operator delete[](void * _Block, const std::nothrow_t &) noexcept; 
#line 105
void __cdecl operator delete(void * _Block, size_t _Size) noexcept; 
#line 110
void __cdecl operator delete[](void * _Block, size_t _Size) noexcept; 
#line 178 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new.h"
#pragma warning(push)
#pragma warning(disable: 4577)
#pragma warning(disable: 4514)
#line 184
inline void *__cdecl operator new(size_t _Size, void *_Where) noexcept 
#line 185
{ 
#line 186
(void)_Size; 
#line 187
return _Where; 
#line 188
} 
#line 190
inline void __cdecl operator delete(void *, void *) noexcept 
#line 191
{ 
#line 193
} 
#line 199 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new.h"
inline void *__cdecl operator new[](size_t _Size, void *
#line 200
_Where) noexcept 
#line 201
{ 
#line 202
(void)_Size; 
#line 203
return _Where; 
#line 204
} 
#line 206
inline void __cdecl operator delete[](void *, void *) noexcept 
#line 207
{ 
#line 208
} 
#line 217 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new.h"
}
#line 210
#pragma warning(pop)
#line 214
#pragma warning(pop)
#pragma pack ( pop )
#line 13 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new_debug.h"
extern "C++" {
#line 15
#pragma pack ( push, 8 )
#line 23
__declspec(allocator) void *__cdecl operator new(size_t _Size, int _BlockUse, const char * _FileName, int _LineNumber); 
#line 31
__declspec(allocator) void *__cdecl operator new[](size_t _Size, int _BlockUse, const char * _FileName, int _LineNumber); 
#line 38
void __cdecl operator delete(void * _Block, int _BlockUse, const char * _FileName, int _LineNumber) noexcept; 
#line 45
void __cdecl operator delete[](void * _Block, int _BlockUse, const char * _FileName, int _LineNumber) noexcept; 
#line 58 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\vcruntime_new_debug.h"
}
#line 56
#pragma pack ( pop )
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\crtdbg.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
typedef void *_HFILE; 
#line 45
typedef int (__cdecl *_CRT_REPORT_HOOK)(int, char *, int *); 
#line 46
typedef int (__cdecl *_CRT_REPORT_HOOKW)(int, __wchar_t *, int *); 
#line 52
typedef int (__cdecl *_CRT_ALLOC_HOOK)(int, void *, size_t, int, long, const unsigned char *, int); 
#line 108
typedef void (__cdecl *_CRT_DUMP_CLIENT)(void *, size_t); 
#line 114
struct _CrtMemBlockHeader; 
#line 123
typedef 
#line 116
struct _CrtMemState { 
#line 118
_CrtMemBlockHeader *pBlockHeader; 
#line 119
size_t lCounts[5]; 
#line 120
size_t lSizes[5]; 
#line 121
size_t lHighWaterCount; 
#line 122
size_t lTotalCount; 
#line 123
} _CrtMemState; 
#line 809 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.17763.0\\ucrt\\crtdbg.h"
}__pragma( pack ( pop )) 
#line 10 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
#pragma pack ( push, 8 )
#pragma warning(push,3)
#pragma warning(disable: 4455 4494 4619 4643 4702 4984 4988 )
#line 132 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
#pragma detect_mismatch("_MSC_VER", "1900")
#line 136 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
#pragma detect_mismatch("_ITERATOR_DEBUG_LEVEL", "0")
#line 141 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
#pragma detect_mismatch("RuntimeLibrary", "MT_StaticRelease")
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\use_ansi.h"
#pragma comment(lib, "libcpmt")
#line 332 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
namespace std { 
#line 333
enum _Uninitialized { 
#line 335
_Noinit
#line 336
}; 
#line 339
class _Lockit { 
#line 360 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
public: __thiscall _Lockit(); 
#line 361
explicit __thiscall _Lockit(int); 
#line 362
__thiscall ~_Lockit() noexcept; 
#line 365 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
static void __cdecl _Lockit_ctor(int); 
#line 366
static void __cdecl _Lockit_dtor(int); 
#line 369
private: static void __cdecl _Lockit_ctor(_Lockit *); 
#line 370
static void __cdecl _Lockit_ctor(_Lockit *, int); 
#line 371
static void __cdecl _Lockit_dtor(_Lockit *); 
#line 374
public: _Lockit(const _Lockit &) = delete;
#line 375
_Lockit &operator=(const _Lockit &) = delete;
#line 378
private: int _Locktype; 
#line 379
}; 
#line 465 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
class _Init_locks { 
#line 480 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
public: __thiscall _Init_locks(); 
#line 481
__thiscall ~_Init_locks() noexcept; 
#line 485 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
private: static void __cdecl _Init_locks_ctor(_Init_locks *); 
#line 486
static void __cdecl _Init_locks_dtor(_Init_locks *); 
#line 487
}; 
#line 524 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
}
#line 533 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\yvals.h"
#pragma warning(pop)
#pragma pack ( pop )
#line 11 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\cstdlib"
#pragma pack ( push, 8 )
#pragma warning(push,3)
#pragma warning(disable: 4455 4494 4619 4643 4702 4984 4988 )
#line 19
inline double abs(double _Xx) noexcept 
#line 20
{ 
#line 21
return ::fabs(_Xx); 
#line 22
} 
#line 24
inline float abs(float _Xx) noexcept 
#line 25
{ 
#line 26
return ::fabsf(_Xx); 
#line 27
} 
#line 29
inline long double abs(long double _Xx) noexcept 
#line 30
{ 
#line 31
return ::fabsl(_Xx); 
#line 32
} 
#line 34
namespace std { 
#line 35
using ::size_t;using ::div_t;using ::ldiv_t;
#line 36
using ::abort;using ::abs;using ::atexit;
#line 37
using ::atof;using ::atoi;using ::atol;
#line 38
using ::bsearch;using ::calloc;using ::div;
#line 39
using ::exit;using ::free;
#line 40
using ::labs;using ::ldiv;using ::malloc;
#line 41
using ::mblen;using ::mbstowcs;using ::mbtowc;
#line 42
using ::qsort;using ::rand;using ::realloc;
#line 43
using ::srand;using ::strtod;using ::strtol;
#line 44
using ::strtoul;
#line 45
using ::wcstombs;using ::wctomb;
#line 47
using ::lldiv_t;
#line 50
using ::getenv;
#line 51
using ::system;
#line 54 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\cstdlib"
using ::atoll;using ::llabs;using ::lldiv;
#line 55
using ::strtof;using ::strtold;
#line 56
using ::strtoll;using ::strtoull;
#line 58
using ::_Exit;using ::at_quick_exit;using ::quick_exit;
#line 59
}
#line 63
#pragma warning(pop)
#pragma pack ( pop )
#line 10 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\cmath"
#pragma pack ( push, 8 )
#pragma warning(push,3)
#pragma warning(disable: 4455 4494 4619 4643 4702 4984 4988 )
#line 17
inline double pow(double _Xx, int _Yx) noexcept 
#line 18
{ 
#line 19
if (_Yx == 2) { 
#line 20
return _Xx * _Xx; }  
#line 22
return ::pow(_Xx, static_cast< double>(_Yx)); 
#line 23
} 
#line 25
inline float acos(float _Xx) noexcept 
#line 26
{ 
#line 27
return ::acosf(_Xx); 
#line 28
} 
#line 30
inline float acosh(float _Xx) noexcept 
#line 31
{ 
#line 32
return ::acoshf(_Xx); 
#line 33
} 
#line 35
inline float asin(float _Xx) noexcept 
#line 36
{ 
#line 37
return ::asinf(_Xx); 
#line 38
} 
#line 40
inline float asinh(float _Xx) noexcept 
#line 41
{ 
#line 42
return ::asinhf(_Xx); 
#line 43
} 
#line 45
inline float atan(float _Xx) noexcept 
#line 46
{ 
#line 47
return ::atanf(_Xx); 
#line 48
} 
#line 50
inline float atanh(float _Xx) noexcept 
#line 51
{ 
#line 52
return ::atanhf(_Xx); 
#line 53
} 
#line 55
inline float atan2(float _Yx, float _Xx) noexcept 
#line 56
{ 
#line 57
return ::atan2f(_Yx, _Xx); 
#line 58
} 
#line 60
inline float cbrt(float _Xx) noexcept 
#line 61
{ 
#line 62
return ::cbrtf(_Xx); 
#line 63
} 
#line 65
inline float ceil(float _Xx) noexcept 
#line 66
{ 
#line 67
return ::ceilf(_Xx); 
#line 68
} 
#line 70
inline float copysign(float _Number, float 
#line 71
_Sign) noexcept 
#line 72
{ 
#line 73
return ::copysignf(_Number, _Sign); 
#line 74
} 
#line 76
inline float cos(float _Xx) noexcept 
#line 77
{ 
#line 78
return ::cosf(_Xx); 
#line 79
} 
#line 81
inline float cosh(float _Xx) noexcept 
#line 82
{ 
#line 83
return ::coshf(_Xx); 
#line 84
} 
#line 86
inline float erf(float _Xx) noexcept 
#line 87
{ 
#line 88
return ::erff(_Xx); 
#line 89
} 
#line 91
inline float erfc(float _Xx) noexcept 
#line 92
{ 
#line 93
return ::erfcf(_Xx); 
#line 94
} 
#line 96
inline float exp(float _Xx) noexcept 
#line 97
{ 
#line 98
return ::expf(_Xx); 
#line 99
} 
#line 101
inline float exp2(float _Xx) noexcept 
#line 102
{ 
#line 103
return ::exp2f(_Xx); 
#line 104
} 
#line 106
inline float expm1(float _Xx) noexcept 
#line 107
{ 
#line 108
return ::expm1f(_Xx); 
#line 109
} 
#line 111
inline float fabs(float _Xx) noexcept 
#line 112
{ 
#line 113
return ::fabsf(_Xx); 
#line 114
} 
#line 116
inline float fdim(float _Xx, float _Yx) noexcept 
#line 117
{ 
#line 118
return ::fdimf(_Xx, _Yx); 
#line 119
} 
#line 121
inline float floor(float _Xx) noexcept 
#line 122
{ 
#line 123
return ::floorf(_Xx); 
#line 124
} 
#line 126
inline float fma(float _Xx, float _Yx, float 
#line 127
_Zx) noexcept 
#line 128
{ 
#line 129
return ::fmaf(_Xx, _Yx, _Zx); 
#line 130
} 
#line 132
inline float fmax(float _Xx, float _Yx) noexcept 
#line 133
{ 
#line 134
return ::fmaxf(_Xx, _Yx); 
#line 135
} 
#line 137
inline float fmin(float _Xx, float _Yx) noexcept 
#line 138
{ 
#line 139
return ::fminf(_Xx, _Yx); 
#line 140
} 
#line 142
inline float fmod(float _Xx, float _Yx) noexcept 
#line 143
{ 
#line 144
return ::fmodf(_Xx, _Yx); 
#line 145
} 
#line 147
inline float frexp(float _Xx, int *_Yx) noexcept 
#line 148
{ 
#line 149
return ::frexpf(_Xx, _Yx); 
#line 150
} 
#line 152
inline float hypot(float _Xx, float _Yx) noexcept 
#line 153
{ 
#line 154
return ::hypotf(_Xx, _Yx); 
#line 155
} 
#line 157
inline int ilogb(float _Xx) noexcept 
#line 158
{ 
#line 159
return ::ilogbf(_Xx); 
#line 160
} 
#line 162
inline float ldexp(float _Xx, int _Yx) noexcept 
#line 163
{ 
#line 164
return ::ldexpf(_Xx, _Yx); 
#line 165
} 
#line 167
inline float lgamma(float _Xx) noexcept 
#line 168
{ 
#line 169
return ::lgammaf(_Xx); 
#line 170
} 
#line 172
inline __int64 llrint(float _Xx) noexcept 
#line 173
{ 
#line 174
return ::llrintf(_Xx); 
#line 175
} 
#line 177
inline __int64 llround(float _Xx) noexcept 
#line 178
{ 
#line 179
return ::llroundf(_Xx); 
#line 180
} 
#line 182
inline float log(float _Xx) noexcept 
#line 183
{ 
#line 184
return ::logf(_Xx); 
#line 185
} 
#line 187
inline float log10(float _Xx) noexcept 
#line 188
{ 
#line 189
return ::log10f(_Xx); 
#line 190
} 
#line 192
inline float log1p(float _Xx) noexcept 
#line 193
{ 
#line 194
return ::log1pf(_Xx); 
#line 195
} 
#line 197
inline float log2(float _Xx) noexcept 
#line 198
{ 
#line 199
return ::log2f(_Xx); 
#line 200
} 
#line 202
inline float logb(float _Xx) noexcept 
#line 203
{ 
#line 204
return ::logbf(_Xx); 
#line 205
} 
#line 207
inline long lrint(float _Xx) noexcept 
#line 208
{ 
#line 209
return ::lrintf(_Xx); 
#line 210
} 
#line 212
inline long lround(float _Xx) noexcept 
#line 213
{ 
#line 214
return ::lroundf(_Xx); 
#line 215
} 
#line 217
inline float modf(float _Xx, float *_Yx) noexcept 
#line 218
{ 
#line 219
return ::modff(_Xx, _Yx); 
#line 220
} 
#line 222
inline float nearbyint(float _Xx) noexcept 
#line 223
{ 
#line 224
return ::nearbyintf(_Xx); 
#line 225
} 
#line 227
inline float nextafter(float _Xx, float _Yx) noexcept 
#line 228
{ 
#line 229
return ::nextafterf(_Xx, _Yx); 
#line 230
} 
#line 232
inline float nexttoward(float _Xx, long double 
#line 233
_Yx) noexcept 
#line 234
{ 
#line 235
return ::nexttowardf(_Xx, _Yx); 
#line 236
} 
#line 238
inline float pow(float _Xx, float 
#line 239
_Yx) noexcept 
#line 240
{ 
#line 241
return ::powf(_Xx, _Yx); 
#line 242
} 
#line 244
inline float pow(float _Xx, int _Yx) noexcept 
#line 245
{ 
#line 246
if (_Yx == 2) { 
#line 247
return _Xx * _Xx; }  
#line 249
return ::powf(_Xx, static_cast< float>(_Yx)); 
#line 250
} 
#line 252
inline float remainder(float _Xx, float _Yx) noexcept 
#line 253
{ 
#line 254
return ::remainderf(_Xx, _Yx); 
#line 255
} 
#line 257
inline float remquo(float _Xx, float _Yx, int *
#line 258
_Zx) noexcept 
#line 259
{ 
#line 260
return ::remquof(_Xx, _Yx, _Zx); 
#line 261
} 
#line 263
inline float rint(float _Xx) noexcept 
#line 264
{ 
#line 265
return ::rintf(_Xx); 
#line 266
} 
#line 268
inline float round(float _Xx) noexcept 
#line 269
{ 
#line 270
return ::roundf(_Xx); 
#line 271
} 
#line 273
inline float scalbln(float _Xx, long _Yx) noexcept 
#line 274
{ 
#line 275
return ::scalblnf(_Xx, _Yx); 
#line 276
} 
#line 278
inline float scalbn(float _Xx, int _Yx) noexcept 
#line 279
{ 
#line 280
return ::scalbnf(_Xx, _Yx); 
#line 281
} 
#line 283
inline float sin(float _Xx) noexcept 
#line 284
{ 
#line 285
return ::sinf(_Xx); 
#line 286
} 
#line 288
inline float sinh(float _Xx) noexcept 
#line 289
{ 
#line 290
return ::sinhf(_Xx); 
#line 291
} 
#line 293
inline float sqrt(float _Xx) noexcept 
#line 294
{ 
#line 295
return ::sqrtf(_Xx); 
#line 296
} 
#line 298
inline float tan(float _Xx) noexcept 
#line 299
{ 
#line 300
return ::tanf(_Xx); 
#line 301
} 
#line 303
inline float tanh(float _Xx) noexcept 
#line 304
{ 
#line 305
return ::tanhf(_Xx); 
#line 306
} 
#line 308
inline float tgamma(float _Xx) noexcept 
#line 309
{ 
#line 310
return ::tgammaf(_Xx); 
#line 311
} 
#line 313
inline float trunc(float _Xx) noexcept 
#line 314
{ 
#line 315
return ::truncf(_Xx); 
#line 316
} 
#line 318
inline long double acos(long double _Xx) noexcept 
#line 319
{ 
#line 320
return ::acosl(_Xx); 
#line 321
} 
#line 323
inline long double acosh(long double _Xx) noexcept 
#line 324
{ 
#line 325
return ::acoshl(_Xx); 
#line 326
} 
#line 328
inline long double asin(long double _Xx) noexcept 
#line 329
{ 
#line 330
return ::asinl(_Xx); 
#line 331
} 
#line 333
inline long double asinh(long double _Xx) noexcept 
#line 334
{ 
#line 335
return ::asinhl(_Xx); 
#line 336
} 
#line 338
inline long double atan(long double _Xx) noexcept 
#line 339
{ 
#line 340
return ::atanl(_Xx); 
#line 341
} 
#line 343
inline long double atanh(long double _Xx) noexcept 
#line 344
{ 
#line 345
return ::atanhl(_Xx); 
#line 346
} 
#line 348
inline long double atan2(long double _Yx, long double 
#line 349
_Xx) noexcept 
#line 350
{ 
#line 351
return ::atan2l(_Yx, _Xx); 
#line 352
} 
#line 354
inline long double cbrt(long double _Xx) noexcept 
#line 355
{ 
#line 356
return ::cbrtl(_Xx); 
#line 357
} 
#line 359
inline long double ceil(long double _Xx) noexcept 
#line 360
{ 
#line 361
return ::ceill(_Xx); 
#line 362
} 
#line 364
inline long double copysign(long double _Number, long double 
#line 365
_Sign) noexcept 
#line 366
{ 
#line 367
return ::copysignl(_Number, _Sign); 
#line 368
} 
#line 370
inline long double cos(long double _Xx) noexcept 
#line 371
{ 
#line 372
return ::cosl(_Xx); 
#line 373
} 
#line 375
inline long double cosh(long double _Xx) noexcept 
#line 376
{ 
#line 377
return ::coshl(_Xx); 
#line 378
} 
#line 380
inline long double erf(long double _Xx) noexcept 
#line 381
{ 
#line 382
return ::erfl(_Xx); 
#line 383
} 
#line 385
inline long double erfc(long double _Xx) noexcept 
#line 386
{ 
#line 387
return ::erfcl(_Xx); 
#line 388
} 
#line 390
inline long double exp(long double _Xx) noexcept 
#line 391
{ 
#line 392
return ::expl(_Xx); 
#line 393
} 
#line 395
inline long double exp2(long double _Xx) noexcept 
#line 396
{ 
#line 397
return ::exp2l(_Xx); 
#line 398
} 
#line 400
inline long double expm1(long double _Xx) noexcept 
#line 401
{ 
#line 402
return ::expm1l(_Xx); 
#line 403
} 
#line 405
inline long double fabs(long double _Xx) noexcept 
#line 406
{ 
#line 407
return ::fabsl(_Xx); 
#line 408
} 
#line 410
inline long double fdim(long double _Xx, long double 
#line 411
_Yx) noexcept 
#line 412
{ 
#line 413
return ::fdiml(_Xx, _Yx); 
#line 414
} 
#line 416
inline long double floor(long double _Xx) noexcept 
#line 417
{ 
#line 418
return ::floorl(_Xx); 
#line 419
} 
#line 421
inline long double fma(long double _Xx, long double 
#line 422
_Yx, long double _Zx) noexcept 
#line 423
{ 
#line 424
return ::fmal(_Xx, _Yx, _Zx); 
#line 425
} 
#line 427
inline long double fmax(long double _Xx, long double 
#line 428
_Yx) noexcept 
#line 429
{ 
#line 430
return ::fmaxl(_Xx, _Yx); 
#line 431
} 
#line 433
inline long double fmin(long double _Xx, long double 
#line 434
_Yx) noexcept 
#line 435
{ 
#line 436
return ::fminl(_Xx, _Yx); 
#line 437
} 
#line 439
inline long double fmod(long double _Xx, long double 
#line 440
_Yx) noexcept 
#line 441
{ 
#line 442
return ::fmodl(_Xx, _Yx); 
#line 443
} 
#line 445
inline long double frexp(long double _Xx, int *
#line 446
_Yx) noexcept 
#line 447
{ 
#line 448
return ::frexpl(_Xx, _Yx); 
#line 449
} 
#line 451
inline long double hypot(long double _Xx, long double 
#line 452
_Yx) noexcept 
#line 453
{ 
#line 454
return ::hypotl(_Xx, _Yx); 
#line 455
} 
#line 457
inline int ilogb(long double _Xx) noexcept 
#line 458
{ 
#line 459
return ::ilogbl(_Xx); 
#line 460
} 
#line 462
inline long double ldexp(long double _Xx, int 
#line 463
_Yx) noexcept 
#line 464
{ 
#line 465
return ::ldexpl(_Xx, _Yx); 
#line 466
} 
#line 468
inline long double lgamma(long double _Xx) noexcept 
#line 469
{ 
#line 470
return ::lgammal(_Xx); 
#line 471
} 
#line 473
inline __int64 llrint(long double _Xx) noexcept 
#line 474
{ 
#line 475
return ::llrintl(_Xx); 
#line 476
} 
#line 478
inline __int64 llround(long double _Xx) noexcept 
#line 479
{ 
#line 480
return ::llroundl(_Xx); 
#line 481
} 
#line 483
inline long double log(long double _Xx) noexcept 
#line 484
{ 
#line 485
return ::logl(_Xx); 
#line 486
} 
#line 488
inline long double log10(long double _Xx) noexcept 
#line 489
{ 
#line 490
return ::log10l(_Xx); 
#line 491
} 
#line 493
inline long double log1p(long double _Xx) noexcept 
#line 494
{ 
#line 495
return ::log1pl(_Xx); 
#line 496
} 
#line 498
inline long double log2(long double _Xx) noexcept 
#line 499
{ 
#line 500
return ::log2l(_Xx); 
#line 501
} 
#line 503
inline long double logb(long double _Xx) noexcept 
#line 504
{ 
#line 505
return ::logbl(_Xx); 
#line 506
} 
#line 508
inline long lrint(long double _Xx) noexcept 
#line 509
{ 
#line 510
return ::lrintl(_Xx); 
#line 511
} 
#line 513
inline long lround(long double _Xx) noexcept 
#line 514
{ 
#line 515
return ::lroundl(_Xx); 
#line 516
} 
#line 518
inline long double modf(long double _Xx, long double *
#line 519
_Yx) noexcept 
#line 520
{ 
#line 521
return ::modfl(_Xx, _Yx); 
#line 522
} 
#line 524
inline long double nearbyint(long double _Xx) noexcept 
#line 525
{ 
#line 526
return ::nearbyintl(_Xx); 
#line 527
} 
#line 529
inline long double nextafter(long double _Xx, long double 
#line 530
_Yx) noexcept 
#line 531
{ 
#line 532
return ::nextafterl(_Xx, _Yx); 
#line 533
} 
#line 535
inline long double nexttoward(long double _Xx, long double 
#line 536
_Yx) noexcept 
#line 537
{ 
#line 538
return ::nexttowardl(_Xx, _Yx); 
#line 539
} 
#line 541
inline long double pow(long double _Xx, long double 
#line 542
_Yx) noexcept 
#line 543
{ 
#line 544
return ::powl(_Xx, _Yx); 
#line 545
} 
#line 547
inline long double pow(long double _Xx, int 
#line 548
_Yx) noexcept 
#line 549
{ 
#line 550
if (_Yx == 2) { 
#line 551
return _Xx * _Xx; }  
#line 553
return ::powl(_Xx, static_cast< long double>(_Yx)); 
#line 554
} 
#line 556
inline long double remainder(long double _Xx, long double 
#line 557
_Yx) noexcept 
#line 558
{ 
#line 559
return ::remainderl(_Xx, _Yx); 
#line 560
} 
#line 562
inline long double remquo(long double _Xx, long double 
#line 563
_Yx, int *_Zx) noexcept 
#line 564
{ 
#line 565
return ::remquol(_Xx, _Yx, _Zx); 
#line 566
} 
#line 568
inline long double rint(long double _Xx) noexcept 
#line 569
{ 
#line 570
return ::rintl(_Xx); 
#line 571
} 
#line 573
inline long double round(long double _Xx) noexcept 
#line 574
{ 
#line 575
return ::roundl(_Xx); 
#line 576
} 
#line 578
inline long double scalbln(long double _Xx, long 
#line 579
_Yx) noexcept 
#line 580
{ 
#line 581
return ::scalblnl(_Xx, _Yx); 
#line 582
} 
#line 584
inline long double scalbn(long double _Xx, int 
#line 585
_Yx) noexcept 
#line 586
{ 
#line 587
return ::scalbnl(_Xx, _Yx); 
#line 588
} 
#line 590
inline long double sin(long double _Xx) noexcept 
#line 591
{ 
#line 592
return ::sinl(_Xx); 
#line 593
} 
#line 595
inline long double sinh(long double _Xx) noexcept 
#line 596
{ 
#line 597
return ::sinhl(_Xx); 
#line 598
} 
#line 600
inline long double sqrt(long double _Xx) noexcept 
#line 601
{ 
#line 602
return ::sqrtl(_Xx); 
#line 603
} 
#line 605
inline long double tan(long double _Xx) noexcept 
#line 606
{ 
#line 607
return ::tanl(_Xx); 
#line 608
} 
#line 610
inline long double tanh(long double _Xx) noexcept 
#line 611
{ 
#line 612
return ::tanhl(_Xx); 
#line 613
} 
#line 615
inline long double tgamma(long double _Xx) noexcept 
#line 616
{ 
#line 617
return ::tgammal(_Xx); 
#line 618
} 
#line 620
inline long double trunc(long double _Xx) noexcept 
#line 621
{ 
#line 622
return ::truncl(_Xx); 
#line 623
} 
#line 8 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\xtr1common"
#pragma pack ( push, 8 )
#pragma warning(push,3)
#pragma warning(disable: 4455 4494 4619 4643 4702 4984 4988 )
#line 15
namespace std { 
#line 17
template < class _Ty,
 _Ty _Val >
 struct integral_constant
 {
 static constexpr _Ty value = _Val;

 using value_type = _Ty;
 using type = integral_constant;

 constexpr operator value_type ( ) const noexcept
  {
  return ( value );
  }

  constexpr value_type operator ( ) ( ) const noexcept
  {
  return ( value );
  }
 };
#line 38
template< bool _Val> using bool_constant = integral_constant< bool, _Val> ; 
#line 41
using true_type = bool_constant< true> ; 
#line 42
using false_type = bool_constant< false> ; 
#line 45
template< bool _Test, class 
#line 46
_Ty = void> 
#line 47
struct enable_if { 
#line 49
}; 
#line 51
template< class _Ty> 
#line 52
struct enable_if< true, _Ty>  { 
#line 54
using type = _Ty; 
#line 55
}; 
#line 57
template< bool _Test, class 
#line 58
_Ty = void> using enable_if_t = typename enable_if< _Test, _Ty> ::type; 
#line 62
template< bool _Test, class 
#line 63
_Ty1, class 
#line 64
_Ty2> 
#line 65
struct conditional { 
#line 67
using type = _Ty2; 
#line 68
}; 
#line 70
template< class _Ty1, class 
#line 71
_Ty2> 
#line 72
struct conditional< true, _Ty1, _Ty2>  { 
#line 74
using type = _Ty1; 
#line 75
}; 
#line 77
template< bool _Test, class 
#line 78
_Ty1, class 
#line 79
_Ty2> using conditional_t = typename conditional< _Test, _Ty1, _Ty2> ::type; 
#line 83
template< class _Ty1, class 
#line 84
_Ty2> 
#line 85
struct is_same : public false_type { 
#line 88
}; 
#line 90
template< class _Ty1> 
#line 91
struct is_same< _Ty1, _Ty1>  : public true_type { 
#line 94
}; 
#line 96
template< class _Ty, class 
#line 97
_Uty> constexpr bool 
#line 98
is_same_v = (is_same< _Ty, _Uty> ::value); 
#line 101
template< class _Ty> 
#line 102
struct remove_const { 
#line 104
using type = _Ty; 
#line 105
}; 
#line 107
template< class _Ty> 
#line 108
struct remove_const< const _Ty>  { 
#line 110
using type = _Ty; 
#line 111
}; 
#line 113
template< class _Ty> using remove_const_t = typename remove_const< _Ty> ::type; 
#line 117
template< class _Ty> 
#line 118
struct remove_volatile { 
#line 120
using type = _Ty; 
#line 121
}; 
#line 123
template< class _Ty> 
#line 124
struct remove_volatile< volatile _Ty>  { 
#line 126
using type = _Ty; 
#line 127
}; 
#line 129
template< class _Ty> using remove_volatile_t = typename remove_volatile< _Ty> ::type; 
#line 133
template< class _Ty> 
#line 134
struct remove_cv { 
#line 136
using type = _Ty; 
#line 137
}; 
#line 139
template< class _Ty> 
#line 140
struct remove_cv< const _Ty>  { 
#line 142
using type = _Ty; 
#line 143
}; 
#line 145
template< class _Ty> 
#line 146
struct remove_cv< volatile _Ty>  { 
#line 148
using type = _Ty; 
#line 149
}; 
#line 151
template< class _Ty> 
#line 152
struct remove_cv< const volatile _Ty>  { 
#line 154
using type = _Ty; 
#line 155
}; 
#line 157
template< class _Ty> using remove_cv_t = typename remove_cv< _Ty> ::type; 
#line 161
template< class _Ty> 
#line 162
struct _Is_integral : public false_type { 
#line 165
}; 
#line 168
template<> struct _Is_integral< bool>  : public true_type { 
#line 171
}; 
#line 174
template<> struct _Is_integral< char>  : public true_type { 
#line 177
}; 
#line 180
template<> struct _Is_integral< unsigned char>  : public true_type { 
#line 183
}; 
#line 186
template<> struct _Is_integral< signed char>  : public true_type { 
#line 189
}; 
#line 193
template<> struct _Is_integral< __wchar_t>  : public true_type { 
#line 196
}; 
#line 200 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\xtr1common"
template<> struct _Is_integral< char16_t>  : public true_type { 
#line 203
}; 
#line 206
template<> struct _Is_integral< char32_t>  : public true_type { 
#line 209
}; 
#line 212
template<> struct _Is_integral< unsigned short>  : public true_type { 
#line 215
}; 
#line 218
template<> struct _Is_integral< short>  : public true_type { 
#line 221
}; 
#line 224
template<> struct _Is_integral< unsigned>  : public true_type { 
#line 227
}; 
#line 230
template<> struct _Is_integral< int>  : public true_type { 
#line 233
}; 
#line 236
template<> struct _Is_integral< unsigned long>  : public true_type { 
#line 239
}; 
#line 242
template<> struct _Is_integral< long>  : public true_type { 
#line 245
}; 
#line 248
template<> struct _Is_integral< unsigned __int64>  : public true_type { 
#line 251
}; 
#line 254
template<> struct _Is_integral< __int64>  : public true_type { 
#line 257
}; 
#line 260
template< class _Ty> 
#line 261
struct is_integral : public _Is_integral< remove_cv_t< _Ty> > ::type { 
#line 264
}; 
#line 266
template< class _Ty> constexpr bool 
#line 267
is_integral_v = (is_integral< _Ty> ::value); 
#line 270
template< class _Ty> 
#line 271
struct _Is_floating_point : public false_type { 
#line 274
}; 
#line 277
template<> struct _Is_floating_point< float>  : public true_type { 
#line 280
}; 
#line 283
template<> struct _Is_floating_point< double>  : public true_type { 
#line 286
}; 
#line 289
template<> struct _Is_floating_point< long double>  : public true_type { 
#line 292
}; 
#line 295
template< class _Ty> 
#line 296
struct is_floating_point : public _Is_floating_point< remove_cv_t< _Ty> > ::type { 
#line 299
}; 
#line 301
template< class _Ty> constexpr bool 
#line 302
is_floating_point_v = (is_floating_point< _Ty> ::value); 
#line 305
template< class _Ty> 
#line 306
struct is_arithmetic : public bool_constant< is_integral_v< _Ty>  || is_floating_point_v< _Ty> >  { 
#line 310
}; 
#line 312
template< class _Ty> constexpr bool 
#line 313
is_arithmetic_v = (is_arithmetic< _Ty> ::value); 
#line 316
template< class _Ty> 
#line 317
struct remove_reference { 
#line 319
using type = _Ty; 
#line 320
}; 
#line 322
template< class _Ty> 
#line 323
struct remove_reference< _Ty &>  { 
#line 325
using type = _Ty; 
#line 326
}; 
#line 328
template< class _Ty> 
#line 329
struct remove_reference< _Ty &&>  { 
#line 331
using type = _Ty; 
#line 332
}; 
#line 334
template< class _Ty> using remove_reference_t = typename remove_reference< _Ty> ::type; 
#line 337
}
#line 340
#pragma warning(pop)
#pragma pack ( pop )
#line 12 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\xtgmath.h"
#pragma pack ( push, 8 )
#pragma warning(push,3)
#pragma warning(disable: 4455 4494 4619 4643 4702 4984 4988 )
#line 19
namespace std { 
#line 20
template< class _Ty1, class 
#line 21
_Ty2> using _Common_float_type_t = conditional_t< is_same_v< _Ty1, long double>  || is_same_v< _Ty2, long double> , long double, conditional_t< is_same_v< _Ty1, float>  && is_same_v< _Ty2, float> , float, double> > ; 
#line 26
}
#line 66
template < class _Ty1,
 class _Ty2,
 class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline
 :: std :: _Common_float_type_t < _Ty1, _Ty2 > pow ( const _Ty1 _Left, const _Ty2 _Right )
 {
 using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >;
 return ( :: pow ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) );
 }
#line 76
extern "C" double __cdecl acos(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double acos ( _Ty _Left ) { return ( :: acos ( static_cast < double > ( _Left ) ) ); }
#line 77
extern "C" double __cdecl asin(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double asin ( _Ty _Left ) { return ( :: asin ( static_cast < double > ( _Left ) ) ); }
#line 78
extern "C" double __cdecl atan(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double atan ( _Ty _Left ) { return ( :: atan ( static_cast < double > ( _Left ) ) ); }
#line 79
extern "C" double __cdecl atan2(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > atan2 ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return ( :: atan2 ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) ); }
#line 80
extern "C" double __cdecl ceil(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double ceil ( _Ty _Left ) { return ( :: ceil ( static_cast < double > ( _Left ) ) ); }
#line 81
extern "C" double __cdecl cos(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double cos ( _Ty _Left ) { return ( :: cos ( static_cast < double > ( _Left ) ) ); }
#line 82
extern "C" double __cdecl cosh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double cosh ( _Ty _Left ) { return ( :: cosh ( static_cast < double > ( _Left ) ) ); }
#line 83
extern "C" double __cdecl exp(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double exp ( _Ty _Left ) { return ( :: exp ( static_cast < double > ( _Left ) ) ); }
#line 85
extern "C" double __cdecl fabs(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double fabs ( _Ty _Left ) { return ( :: fabs ( static_cast < double > ( _Left ) ) ); }
#line 87
extern "C" double __cdecl floor(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double floor ( _Ty _Left ) { return ( :: floor ( static_cast < double > ( _Left ) ) ); }
#line 88
extern "C" double __cdecl fmod(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmod ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return ( :: fmod ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) ); }
#line 89
extern "C" double __cdecl frexp(double, int *); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double frexp ( _Ty _Left, int * _Arg2 ) { return ( :: frexp ( static_cast < double > ( _Left ), _Arg2 ) ); }
#line 90
extern "C" double __cdecl ldexp(double, int); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double ldexp ( _Ty _Left, int _Arg2 ) { return ( :: ldexp ( static_cast < double > ( _Left ), _Arg2 ) ); }
#line 91
extern "C" double __cdecl log(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double log ( _Ty _Left ) { return ( :: log ( static_cast < double > ( _Left ) ) ); }
#line 92
extern "C" double __cdecl log10(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double log10 ( _Ty _Left ) { return ( :: log10 ( static_cast < double > ( _Left ) ) ); }
#line 95
extern "C" double __cdecl sin(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double sin ( _Ty _Left ) { return ( :: sin ( static_cast < double > ( _Left ) ) ); }
#line 96
extern "C" double __cdecl sinh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double sinh ( _Ty _Left ) { return ( :: sinh ( static_cast < double > ( _Left ) ) ); }
#line 97
extern "C" double __cdecl sqrt(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double sqrt ( _Ty _Left ) { return ( :: sqrt ( static_cast < double > ( _Left ) ) ); }
#line 98
extern "C" double __cdecl tan(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double tan ( _Ty _Left ) { return ( :: tan ( static_cast < double > ( _Left ) ) ); }
#line 99
extern "C" double __cdecl tanh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double tanh ( _Ty _Left ) { return ( :: tanh ( static_cast < double > ( _Left ) ) ); }
#line 105
inline float _Fma(float _Left, float _Middle, float _Right) 
#line 106
{ 
#line 107
return ::fmaf(_Left, _Middle, _Right); 
#line 108
} 
#line 110
inline double _Fma(double _Left, double _Middle, double _Right) 
#line 111
{ 
#line 112
return ::fma(_Left, _Middle, _Right); 
#line 113
} 
#line 115
inline long double _Fma(long double _Left, long double _Middle, long double 
#line 116
_Right) 
#line 117
{ 
#line 118
return ::fmal(_Left, _Middle, _Right); 
#line 119
} 
#line 122 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\xtgmath.h"
template < class _Ty1,
 class _Ty2,
 class _Ty3,
 class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >
  && :: std :: is_arithmetic_v < _Ty3 > > > inline
 :: std :: _Common_float_type_t < _Ty1, :: std :: _Common_float_type_t < _Ty2, _Ty3 > >
 fma ( _Ty1 _Left, _Ty2 _Middle, _Ty3 _Right )
 {
 using _Common = :: std :: _Common_float_type_t < _Ty1, :: std :: _Common_float_type_t < _Ty2, _Ty3 >>;














 return ( _Fma ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Middle ), static_cast < _Common > ( _Right ) ) );

 }
#line 151 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\xtgmath.h"
inline float _Remquo(float _Left, float _Right, int *_Pquo) 
#line 152
{ 
#line 153
return ::remquof(_Left, _Right, _Pquo); 
#line 154
} 
#line 156
inline double _Remquo(double _Left, double _Right, int *_Pquo) 
#line 157
{ 
#line 158
return ::remquo(_Left, _Right, _Pquo); 
#line 159
} 
#line 161
inline long double _Remquo(long double _Left, long double _Right, int *_Pquo) 
#line 162
{ 
#line 163
return ::remquol(_Left, _Right, _Pquo); 
#line 164
} 
#line 167 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\xtgmath.h"
template < class _Ty1,
 class _Ty2,
 class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline
 :: std :: _Common_float_type_t < _Ty1, _Ty2 >
 remquo ( _Ty1 _Left, _Ty2 _Right, int * _Pquo )
 {
 using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >;














 return ( _Remquo ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ), _Pquo ) );

 }
#line 192 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\xtgmath.h"
extern "C" double __cdecl acosh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double acosh ( _Ty _Left ) { return ( :: acosh ( static_cast < double > ( _Left ) ) ); }
#line 193
extern "C" double __cdecl asinh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double asinh ( _Ty _Left ) { return ( :: asinh ( static_cast < double > ( _Left ) ) ); }
#line 194
extern "C" double __cdecl atanh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double atanh ( _Ty _Left ) { return ( :: atanh ( static_cast < double > ( _Left ) ) ); }
#line 195
extern "C" double __cdecl cbrt(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double cbrt ( _Ty _Left ) { return ( :: cbrt ( static_cast < double > ( _Left ) ) ); }
#line 196
extern "C" double __cdecl copysign(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > copysign ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return ( :: copysign ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) ); }
#line 197
extern "C" double __cdecl erf(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double erf ( _Ty _Left ) { return ( :: erf ( static_cast < double > ( _Left ) ) ); }
#line 198
extern "C" double __cdecl erfc(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double erfc ( _Ty _Left ) { return ( :: erfc ( static_cast < double > ( _Left ) ) ); }
#line 199
extern "C" double __cdecl expm1(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double expm1 ( _Ty _Left ) { return ( :: expm1 ( static_cast < double > ( _Left ) ) ); }
#line 200
extern "C" double __cdecl exp2(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double exp2 ( _Ty _Left ) { return ( :: exp2 ( static_cast < double > ( _Left ) ) ); }
#line 201
extern "C" double __cdecl fdim(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > fdim ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return ( :: fdim ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) ); }
#line 203
extern "C" double __cdecl fmax(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmax ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return ( :: fmax ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) ); }
#line 204
extern "C" double __cdecl fmin(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmin ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return ( :: fmin ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) ); }
#line 205
extern "C" double __cdecl hypot(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > hypot ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return ( :: hypot ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) ); }
#line 206
extern "C" int __cdecl ilogb(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline int ilogb ( _Ty _Left ) { return ( :: ilogb ( static_cast < double > ( _Left ) ) ); }
#line 207
extern "C" double __cdecl lgamma(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double lgamma ( _Ty _Left ) { return ( :: lgamma ( static_cast < double > ( _Left ) ) ); }
#line 208
extern "C" __int64 __cdecl llrint(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline long long llrint ( _Ty _Left ) { return ( :: llrint ( static_cast < double > ( _Left ) ) ); }
#line 209
extern "C" __int64 __cdecl llround(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline long long llround ( _Ty _Left ) { return ( :: llround ( static_cast < double > ( _Left ) ) ); }
#line 210
extern "C" double __cdecl log1p(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double log1p ( _Ty _Left ) { return ( :: log1p ( static_cast < double > ( _Left ) ) ); }
#line 211
extern "C" double __cdecl log2(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double log2 ( _Ty _Left ) { return ( :: log2 ( static_cast < double > ( _Left ) ) ); }
#line 212
extern "C" double __cdecl logb(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double logb ( _Ty _Left ) { return ( :: logb ( static_cast < double > ( _Left ) ) ); }
#line 213
extern "C" long __cdecl lrint(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline long lrint ( _Ty _Left ) { return ( :: lrint ( static_cast < double > ( _Left ) ) ); }
#line 214
extern "C" long __cdecl lround(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline long lround ( _Ty _Left ) { return ( :: lround ( static_cast < double > ( _Left ) ) ); }
#line 215
extern "C" double __cdecl nearbyint(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double nearbyint ( _Ty _Left ) { return ( :: nearbyint ( static_cast < double > ( _Left ) ) ); }
#line 216
extern "C" double __cdecl nextafter(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > nextafter ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return ( :: nextafter ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) ); }
#line 217
extern "C" double __cdecl nexttoward(double, long double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double nexttoward ( _Ty _Left, long double _Arg2 ) { return ( :: nexttoward ( static_cast < double > ( _Left ), _Arg2 ) ); }
#line 218
extern "C" double __cdecl remainder(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > remainder ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return ( :: remainder ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ) ); }
#line 220
extern "C" double __cdecl rint(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double rint ( _Ty _Left ) { return ( :: rint ( static_cast < double > ( _Left ) ) ); }
#line 221
extern "C" double __cdecl round(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double round ( _Ty _Left ) { return ( :: round ( static_cast < double > ( _Left ) ) ); }
#line 222
extern "C" double __cdecl scalbln(double, long); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double scalbln ( _Ty _Left, long _Arg2 ) { return ( :: scalbln ( static_cast < double > ( _Left ), _Arg2 ) ); }
#line 223
extern "C" double __cdecl scalbn(double, int); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double scalbn ( _Ty _Left, int _Arg2 ) { return ( :: scalbn ( static_cast < double > ( _Left ), _Arg2 ) ); }
#line 224
extern "C" double __cdecl tgamma(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double tgamma ( _Ty _Left ) { return ( :: tgamma ( static_cast < double > ( _Left ) ) ); }
#line 225
extern "C" double __cdecl trunc(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double trunc ( _Ty _Left ) { return ( :: trunc ( static_cast < double > ( _Left ) ) ); }
#line 237
#pragma warning(pop)
#pragma pack ( pop )
#line 627 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\cmath"
namespace std { 
#line 628
using ::abs;using ::acos;using ::asin;
#line 629
using ::atan;using ::atan2;using ::ceil;
#line 630
using ::cos;using ::cosh;using ::exp;
#line 631
using ::fabs;using ::floor;using ::fmod;
#line 632
using ::frexp;using ::ldexp;using ::log;
#line 633
using ::log10;using ::modf;using ::pow;
#line 634
using ::sin;using ::sinh;using ::sqrt;
#line 635
using ::tan;using ::tanh;
#line 637
using ::acosf;using ::asinf;
#line 638
using ::atanf;using ::atan2f;using ::ceilf;
#line 639
using ::cosf;using ::coshf;using ::expf;
#line 640
using ::fabsf;using ::floorf;using ::fmodf;
#line 641
using ::frexpf;using ::ldexpf;using ::logf;
#line 642
using ::log10f;using ::modff;using ::powf;
#line 643
using ::sinf;using ::sinhf;using ::sqrtf;
#line 644
using ::tanf;using ::tanhf;
#line 646
using ::acosl;using ::asinl;
#line 647
using ::atanl;using ::atan2l;using ::ceill;
#line 648
using ::cosl;using ::coshl;using ::expl;
#line 649
using ::fabsl;using ::floorl;using ::fmodl;
#line 650
using ::frexpl;using ::ldexpl;using ::logl;
#line 651
using ::log10l;using ::modfl;using ::powl;
#line 652
using ::sinl;using ::sinhl;using ::sqrtl;
#line 653
using ::tanl;using ::tanhl;
#line 655
using ::float_t;using ::double_t;
#line 657
using ::acosh;using ::asinh;using ::atanh;
#line 658
using ::cbrt;using ::erf;using ::erfc;
#line 659
using ::expm1;using ::exp2;
#line 660
using ::hypot;using ::ilogb;using ::lgamma;
#line 661
using ::log1p;using ::log2;using ::logb;
#line 662
using ::llrint;using ::lrint;using ::nearbyint;
#line 663
using ::rint;using ::llround;using ::lround;
#line 664
using ::fdim;using ::fma;using ::fmax;using ::fmin;
#line 665
using ::round;using ::trunc;
#line 666
using ::remainder;using ::remquo;
#line 667
using ::copysign;using ::nan;using ::nextafter;
#line 668
using ::scalbn;using ::scalbln;
#line 669
using ::nexttoward;using ::tgamma;
#line 671
using ::acoshf;using ::asinhf;using ::atanhf;
#line 672
using ::cbrtf;using ::erff;using ::erfcf;
#line 673
using ::expm1f;using ::exp2f;
#line 674
using ::hypotf;using ::ilogbf;using ::lgammaf;
#line 675
using ::log1pf;using ::log2f;using ::logbf;
#line 676
using ::llrintf;using ::lrintf;using ::nearbyintf;
#line 677
using ::rintf;using ::llroundf;using ::lroundf;
#line 678
using ::fdimf;using ::fmaf;using ::fmaxf;using ::fminf;
#line 679
using ::roundf;using ::truncf;
#line 680
using ::remainderf;using ::remquof;
#line 681
using ::copysignf;using ::nanf;
#line 682
using ::nextafterf;using ::scalbnf;using ::scalblnf;
#line 683
using ::nexttowardf;using ::tgammaf;
#line 685
using ::acoshl;using ::asinhl;using ::atanhl;
#line 686
using ::cbrtl;using ::erfl;using ::erfcl;
#line 687
using ::expm1l;using ::exp2l;
#line 688
using ::hypotl;using ::ilogbl;using ::lgammal;
#line 689
using ::log1pl;using ::log2l;using ::logbl;
#line 690
using ::llrintl;using ::lrintl;using ::nearbyintl;
#line 691
using ::rintl;using ::llroundl;using ::lroundl;
#line 692
using ::fdiml;using ::fmal;using ::fmaxl;using ::fminl;
#line 693
using ::roundl;using ::truncl;
#line 694
using ::remainderl;using ::remquol;
#line 695
using ::copysignl;using ::nanl;
#line 696
using ::nextafterl;using ::scalbnl;using ::scalblnl;
#line 697
using ::nexttowardl;using ::tgammal;
#line 699
using ::fpclassify;using ::signbit;
#line 700
using ::isfinite;using ::isinf;
#line 701
using ::isnan;using ::isnormal;
#line 702
using ::isgreater;using ::isgreaterequal;
#line 703
using ::isless;using ::islessequal;
#line 704
using ::islessgreater;using ::isunordered;
#line 705
}
#line 1183 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Tools\\MSVC\\14.16.27023\\include\\cmath"
#pragma warning(pop)
#pragma pack ( pop )
#line 9087 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern "C" double __cdecl _hypot(double x, double y); 
#line 9088
extern "C" float __cdecl _hypotf(float x, float y); 
#line 9098 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline bool signbit(long double) throw(); 
#line 9099
extern "C" int _ldsign(long double); 
#line 9142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline bool signbit(double) throw(); 
#line 9143
extern "C" int _dsign(double); 
#line 9187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline bool signbit(float) throw(); 
#line 9188
extern "C" int _fdsign(float); 
#line 9196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline bool isinf(long double a); 
#line 9231 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline bool isinf(double a); 
#line 9269 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline bool isinf(float a); 
#line 9276 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline bool isnan(long double a); 
#line 9309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline bool isnan(double a); 
#line 9345 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline bool isnan(float a); 
#line 9352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline bool isfinite(long double a); 
#line 9389 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline bool isfinite(double a); 
#line 9425 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static __inline bool isfinite(float a); 
#line 9433 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
template< class T> extern T _Pow_int(T, int) throw(); 
#line 9434
extern inline __int64 abs(__int64) throw(); 
#line 9509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline long __cdecl abs(long) throw(); 
#line 9513 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline float __cdecl abs(float) throw(); 
#line 9514
extern inline double __cdecl abs(double) throw(); 
#line 9515
extern inline float __cdecl fabs(float) throw(); 
#line 9516
extern inline float __cdecl ceil(float) throw(); 
#line 9517
extern inline float __cdecl floor(float) throw(); 
#line 9518
extern inline float __cdecl sqrt(float) throw(); 
#line 9519
extern inline float __cdecl pow(float, float) throw(); 
#line 9544 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline float __cdecl pow(float, int) throw(); 
#line 9545
extern inline double __cdecl pow(double, int) throw(); 
#line 9548 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline float __cdecl log(float) throw(); 
#line 9549
extern inline float __cdecl log10(float) throw(); 
#line 9550
extern inline float __cdecl fmod(float, float) throw(); 
#line 9551
extern inline float __cdecl modf(float, float *) throw(); 
#line 9552
extern inline float __cdecl exp(float) throw(); 
#line 9553
extern inline float __cdecl frexp(float, int *) throw(); 
#line 9554
extern inline float __cdecl ldexp(float, int) throw(); 
#line 9555
extern inline float __cdecl asin(float) throw(); 
#line 9556
extern inline float __cdecl sin(float) throw(); 
#line 9557
extern inline float __cdecl sinh(float) throw(); 
#line 9558
extern inline float __cdecl acos(float) throw(); 
#line 9559
extern inline float __cdecl cos(float) throw(); 
#line 9560
extern inline float __cdecl cosh(float) throw(); 
#line 9561
extern inline float __cdecl atan(float) throw(); 
#line 9562
extern inline float __cdecl atan2(float, float) throw(); 
#line 9563
extern inline float __cdecl tan(float) throw(); 
#line 9564
extern inline float __cdecl tanh(float) throw(); 
#line 9794 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
extern inline float __cdecl logb(float) throw(); 
#line 9795
extern inline int __cdecl ilogb(float) throw(); 
#line 9796
extern float __cdecl scalbn(float, float) throw(); 
#line 9797
extern inline float __cdecl scalbln(float, long) throw(); 
#line 9798
extern inline float __cdecl exp2(float) throw(); 
#line 9799
extern inline float __cdecl expm1(float) throw(); 
#line 9800
extern inline float __cdecl log2(float) throw(); 
#line 9801
extern inline float __cdecl log1p(float) throw(); 
#line 9802
extern inline float __cdecl acosh(float) throw(); 
#line 9803
extern inline float __cdecl asinh(float) throw(); 
#line 9804
extern inline float __cdecl atanh(float) throw(); 
#line 9805
extern inline float __cdecl hypot(float, float) throw(); 
#line 9806
extern float __cdecl norm3d(float, float, float) throw(); 
#line 9807
extern float __cdecl norm4d(float, float, float, float) throw(); 
#line 9808
extern inline float __cdecl cbrt(float) throw(); 
#line 9809
extern inline float __cdecl erf(float) throw(); 
#line 9810
extern inline float __cdecl erfc(float) throw(); 
#line 9811
extern inline float __cdecl lgamma(float) throw(); 
#line 9812
extern inline float __cdecl tgamma(float) throw(); 
#line 9813
extern inline float __cdecl copysign(float, float) throw(); 
#line 9814
extern inline float __cdecl nextafter(float, float) throw(); 
#line 9815
extern inline float __cdecl remainder(float, float) throw(); 
#line 9816
extern inline float __cdecl remquo(float, float, int *) throw(); 
#line 9817
extern inline float __cdecl round(float) throw(); 
#line 9818
extern inline long __cdecl lround(float) throw(); 
#line 9819
extern inline __int64 __cdecl llround(float) throw(); 
#line 9820
extern inline float __cdecl trunc(float) throw(); 
#line 9821
extern inline float __cdecl rint(float) throw(); 
#line 9822
extern inline long __cdecl lrint(float) throw(); 
#line 9823
extern inline __int64 __cdecl llrint(float) throw(); 
#line 9824
extern inline float __cdecl nearbyint(float) throw(); 
#line 9825
extern inline float __cdecl fdim(float, float) throw(); 
#line 9826
extern inline float __cdecl fma(float, float, float) throw(); 
#line 9827
extern inline float __cdecl fmax(float, float) throw(); 
#line 9828
extern inline float __cdecl fmin(float, float) throw(); 
#line 9831 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.h"
static inline float exp10(float a); 
#line 9833
static inline float rsqrt(float a); 
#line 9835
static inline float rcbrt(float a); 
#line 9837
static inline float sinpi(float a); 
#line 9839
static inline float cospi(float a); 
#line 9841
static inline void sincospi(float a, float * sptr, float * cptr); 
#line 9843
static inline void sincos(float a, float * sptr, float * cptr); 
#line 9845
static inline float j0(float a); 
#line 9847
static inline float j1(float a); 
#line 9849
static inline float jn(int n, float a); 
#line 9851
static inline float y0(float a); 
#line 9853
static inline float y1(float a); 
#line 9855
static inline float yn(int n, float a); 
#line 9857
static inline float cyl_bessel_i0(float a); 
#line 9859
static inline float cyl_bessel_i1(float a); 
#line 9861
static inline float erfinv(float a); 
#line 9863
static inline float erfcinv(float a); 
#line 9865
static inline float normcdfinv(float a); 
#line 9867
static inline float normcdf(float a); 
#line 9869
static inline float erfcx(float a); 
#line 9871
static inline double copysign(double a, float b); 
#line 9873
static inline double copysign(float a, double b); 
#line 9875
static inline unsigned min(unsigned a, unsigned b); 
#line 9877
static inline unsigned min(int a, unsigned b); 
#line 9879
static inline unsigned min(unsigned a, int b); 
#line 9881
static inline long min(long a, long b); 
#line 9883
static inline unsigned long min(unsigned long a, unsigned long b); 
#line 9885
static inline unsigned long min(long a, unsigned long b); 
#line 9887
static inline unsigned long min(unsigned long a, long b); 
#line 9889
static inline __int64 min(__int64 a, __int64 b); 
#line 9891
static inline unsigned __int64 min(unsigned __int64 a, unsigned __int64 b); 
#line 9893
static inline unsigned __int64 min(__int64 a, unsigned __int64 b); 
#line 9895
static inline unsigned __int64 min(unsigned __int64 a, __int64 b); 
#line 9897
static inline float min(float a, float b); 
#line 9899
static inline double min(double a, double b); 
#line 9901
static inline double min(float a, double b); 
#line 9903
static inline double min(double a, float b); 
#line 9905
static inline unsigned max(unsigned a, unsigned b); 
#line 9907
static inline unsigned max(int a, unsigned b); 
#line 9909
static inline unsigned max(unsigned a, int b); 
#line 9911
static inline long max(long a, long b); 
#line 9913
static inline unsigned long max(unsigned long a, unsigned long b); 
#line 9915
static inline unsigned long max(long a, unsigned long b); 
#line 9917
static inline unsigned long max(unsigned long a, long b); 
#line 9919
static inline __int64 max(__int64 a, __int64 b); 
#line 9921
static inline unsigned __int64 max(unsigned __int64 a, unsigned __int64 b); 
#line 9923
static inline unsigned __int64 max(__int64 a, unsigned __int64 b); 
#line 9925
static inline unsigned __int64 max(unsigned __int64 a, __int64 b); 
#line 9927
static inline float max(float a, float b); 
#line 9929
static inline double max(double a, double b); 
#line 9931
static inline double max(float a, double b); 
#line 9933
static inline double max(double a, float b); 
#line 422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static __inline bool isinf(long double a) 
#line 423
{ 
#line 427 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return isinf< long double> (a); 
#line 429 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
} 
#line 438 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static __inline bool isinf(double a) 
#line 439
{ 
#line 443 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return isinf< double> (a); 
#line 445 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
} 
#line 454 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static __inline bool isinf(float a) 
#line 455
{ 
#line 459 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return isinf< float> (a); 
#line 461 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
} 
#line 470 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static __inline bool isnan(long double a) 
#line 471
{ 
#line 475 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return isnan< long double> (a); 
#line 477 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
} 
#line 486 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static __inline bool isnan(double a) 
#line 487
{ 
#line 491 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return isnan< double> (a); 
#line 493 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
} 
#line 502 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static __inline bool isnan(float a) 
#line 503
{ 
#line 507 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return isnan< float> (a); 
#line 509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
} 
#line 518 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(long double a) 
#line 519
{ 
#line 523 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return isfinite< long double> (a); 
#line 525 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
} 
#line 534 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(double a) 
#line 535
{ 
#line 539 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return isfinite< double> (a); 
#line 541 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
} 
#line 550 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(float a) 
#line 551
{ 
#line 555 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return isfinite< float> (a); 
#line 557 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
} 
#line 765 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static inline float exp10(float a) 
#line 766
{ 
#line 767
return exp10f(a); 
#line 768
} 
#line 770
static inline float rsqrt(float a) 
#line 771
{ 
#line 772
return rsqrtf(a); 
#line 773
} 
#line 775
static inline float rcbrt(float a) 
#line 776
{ 
#line 777
return rcbrtf(a); 
#line 778
} 
#line 780
static inline float sinpi(float a) 
#line 781
{ 
#line 782
return sinpif(a); 
#line 783
} 
#line 785
static inline float cospi(float a) 
#line 786
{ 
#line 787
return cospif(a); 
#line 788
} 
#line 790
static inline void sincospi(float a, float *sptr, float *cptr) 
#line 791
{ 
#line 792
sincospif(a, sptr, cptr); 
#line 793
} 
#line 795
static inline void sincos(float a, float *sptr, float *cptr) 
#line 796
{ 
#line 797
sincosf(a, sptr, cptr); 
#line 798
} 
#line 800
static inline float j0(float a) 
#line 801
{ 
#line 802
return j0f(a); 
#line 803
} 
#line 805
static inline float j1(float a) 
#line 806
{ 
#line 807
return j1f(a); 
#line 808
} 
#line 810
static inline float jn(int n, float a) 
#line 811
{ 
#line 812
return jnf(n, a); 
#line 813
} 
#line 815
static inline float y0(float a) 
#line 816
{ 
#line 817
return y0f(a); 
#line 818
} 
#line 820
static inline float y1(float a) 
#line 821
{ 
#line 822
return y1f(a); 
#line 823
} 
#line 825
static inline float yn(int n, float a) 
#line 826
{ 
#line 827
return ynf(n, a); 
#line 828
} 
#line 830
static inline float cyl_bessel_i0(float a) 
#line 831
{ 
#line 832
return cyl_bessel_i0f(a); 
#line 833
} 
#line 835
static inline float cyl_bessel_i1(float a) 
#line 836
{ 
#line 837
return cyl_bessel_i1f(a); 
#line 838
} 
#line 840
static inline float erfinv(float a) 
#line 841
{ 
#line 842
return erfinvf(a); 
#line 843
} 
#line 845
static inline float erfcinv(float a) 
#line 846
{ 
#line 847
return erfcinvf(a); 
#line 848
} 
#line 850
static inline float normcdfinv(float a) 
#line 851
{ 
#line 852
return normcdfinvf(a); 
#line 853
} 
#line 855
static inline float normcdf(float a) 
#line 856
{ 
#line 857
return normcdff(a); 
#line 858
} 
#line 860
static inline float erfcx(float a) 
#line 861
{ 
#line 862
return erfcxf(a); 
#line 863
} 
#line 865
static inline double copysign(double a, float b) 
#line 866
{ 
#line 867
return copysign(a, (double)b); 
#line 868
} 
#line 870
static inline double copysign(float a, double b) 
#line 871
{ 
#line 872
return copysign((double)a, b); 
#line 873
} 
#line 875
static inline unsigned min(unsigned a, unsigned b) 
#line 876
{ 
#line 877
return umin(a, b); 
#line 878
} 
#line 880
static inline unsigned min(int a, unsigned b) 
#line 881
{ 
#line 882
return umin((unsigned)a, b); 
#line 883
} 
#line 885
static inline unsigned min(unsigned a, int b) 
#line 886
{ 
#line 887
return umin(a, (unsigned)b); 
#line 888
} 
#line 890
static inline long min(long a, long b) 
#line 891
{ 
#line 894
#pragma warning (disable: 4127)
#line 897 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 899
#pragma warning (default: 4127)
#line 901 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return (long)min((int)a, (int)b); 
#line 902
} else { 
#line 903
return (long)llmin((__int64)a, (__int64)b); 
#line 904
}  
#line 905
} 
#line 907
static inline unsigned long min(unsigned long a, unsigned long b) 
#line 908
{ 
#line 910
#pragma warning (disable: 4127)
#line 912 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 914
#pragma warning (default: 4127)
#line 916 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return (unsigned long)umin((unsigned)a, (unsigned)b); 
#line 917
} else { 
#line 918
return (unsigned long)ullmin((unsigned __int64)a, (unsigned __int64)b); 
#line 919
}  
#line 920
} 
#line 922
static inline unsigned long min(long a, unsigned long b) 
#line 923
{ 
#line 925
#pragma warning (disable: 4127)
#line 927 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 929
#pragma warning (default: 4127)
#line 931 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return (unsigned long)umin((unsigned)a, (unsigned)b); 
#line 932
} else { 
#line 933
return (unsigned long)ullmin((unsigned __int64)a, (unsigned __int64)b); 
#line 934
}  
#line 935
} 
#line 937
static inline unsigned long min(unsigned long a, long b) 
#line 938
{ 
#line 940
#pragma warning (disable: 4127)
#line 942 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 944
#pragma warning (default: 4127)
#line 946 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return (unsigned long)umin((unsigned)a, (unsigned)b); 
#line 947
} else { 
#line 948
return (unsigned long)ullmin((unsigned __int64)a, (unsigned __int64)b); 
#line 949
}  
#line 950
} 
#line 952
static inline __int64 min(__int64 a, __int64 b) 
#line 953
{ 
#line 954
return llmin(a, b); 
#line 955
} 
#line 957
static inline unsigned __int64 min(unsigned __int64 a, unsigned __int64 b) 
#line 958
{ 
#line 959
return ullmin(a, b); 
#line 960
} 
#line 962
static inline unsigned __int64 min(__int64 a, unsigned __int64 b) 
#line 963
{ 
#line 964
return ullmin((unsigned __int64)a, b); 
#line 965
} 
#line 967
static inline unsigned __int64 min(unsigned __int64 a, __int64 b) 
#line 968
{ 
#line 969
return ullmin(a, (unsigned __int64)b); 
#line 970
} 
#line 972
static inline float min(float a, float b) 
#line 973
{ 
#line 974
return fminf(a, b); 
#line 975
} 
#line 977
static inline double min(double a, double b) 
#line 978
{ 
#line 979
return fmin(a, b); 
#line 980
} 
#line 982
static inline double min(float a, double b) 
#line 983
{ 
#line 984
return fmin((double)a, b); 
#line 985
} 
#line 987
static inline double min(double a, float b) 
#line 988
{ 
#line 989
return fmin(a, (double)b); 
#line 990
} 
#line 992
static inline unsigned max(unsigned a, unsigned b) 
#line 993
{ 
#line 994
return umax(a, b); 
#line 995
} 
#line 997
static inline unsigned max(int a, unsigned b) 
#line 998
{ 
#line 999
return umax((unsigned)a, b); 
#line 1000
} 
#line 1002
static inline unsigned max(unsigned a, int b) 
#line 1003
{ 
#line 1004
return umax(a, (unsigned)b); 
#line 1005
} 
#line 1007
static inline long max(long a, long b) 
#line 1008
{ 
#line 1011
#pragma warning (disable: 4127)
#line 1013 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 1015
#pragma warning (default: 4127)
#line 1017 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return (long)max((int)a, (int)b); 
#line 1018
} else { 
#line 1019
return (long)llmax((__int64)a, (__int64)b); 
#line 1020
}  
#line 1021
} 
#line 1023
static inline unsigned long max(unsigned long a, unsigned long b) 
#line 1024
{ 
#line 1026
#pragma warning (disable: 4127)
#line 1028 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1030
#pragma warning (default: 4127)
#line 1032 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return (unsigned long)umax((unsigned)a, (unsigned)b); 
#line 1033
} else { 
#line 1034
return (unsigned long)ullmax((unsigned __int64)a, (unsigned __int64)b); 
#line 1035
}  
#line 1036
} 
#line 1038
static inline unsigned long max(long a, unsigned long b) 
#line 1039
{ 
#line 1041
#pragma warning (disable: 4127)
#line 1043 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1045
#pragma warning (default: 4127)
#line 1047 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return (unsigned long)umax((unsigned)a, (unsigned)b); 
#line 1048
} else { 
#line 1049
return (unsigned long)ullmax((unsigned __int64)a, (unsigned __int64)b); 
#line 1050
}  
#line 1051
} 
#line 1053
static inline unsigned long max(unsigned long a, long b) 
#line 1054
{ 
#line 1056
#pragma warning (disable: 4127)
#line 1058 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1060
#pragma warning (default: 4127)
#line 1062 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
return (unsigned long)umax((unsigned)a, (unsigned)b); 
#line 1063
} else { 
#line 1064
return (unsigned long)ullmax((unsigned __int64)a, (unsigned __int64)b); 
#line 1065
}  
#line 1066
} 
#line 1068
static inline __int64 max(__int64 a, __int64 b) 
#line 1069
{ 
#line 1070
return llmax(a, b); 
#line 1071
} 
#line 1073
static inline unsigned __int64 max(unsigned __int64 a, unsigned __int64 b) 
#line 1074
{ 
#line 1075
return ullmax(a, b); 
#line 1076
} 
#line 1078
static inline unsigned __int64 max(__int64 a, unsigned __int64 b) 
#line 1079
{ 
#line 1080
return ullmax((unsigned __int64)a, b); 
#line 1081
} 
#line 1083
static inline unsigned __int64 max(unsigned __int64 a, __int64 b) 
#line 1084
{ 
#line 1085
return ullmax(a, (unsigned __int64)b); 
#line 1086
} 
#line 1088
static inline float max(float a, float b) 
#line 1089
{ 
#line 1090
return fmaxf(a, b); 
#line 1091
} 
#line 1093
static inline double max(double a, double b) 
#line 1094
{ 
#line 1095
return fmax(a, b); 
#line 1096
} 
#line 1098
static inline double max(float a, double b) 
#line 1099
{ 
#line 1100
return fmax((double)a, b); 
#line 1101
} 
#line 1103
static inline double max(double a, float b) 
#line 1104
{ 
#line 1105
return fmax(a, (double)b); 
#line 1106
} 
#line 1112
#pragma warning(disable : 4211)
#line 1117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\math_functions.hpp"
static inline int min(int a, int b) 
#line 1118
{ 
#line 1119
return (a < b) ? a : b; 
#line 1120
} 
#line 1122
static inline unsigned umin(unsigned a, unsigned b) 
#line 1123
{ 
#line 1124
return (a < b) ? a : b; 
#line 1125
} 
#line 1127
static inline __int64 llmin(__int64 a, __int64 b) 
#line 1128
{ 
#line 1129
return (a < b) ? a : b; 
#line 1130
} 
#line 1132
static inline unsigned __int64 ullmin(unsigned __int64 a, unsigned __int64 
#line 1133
b) 
#line 1134
{ 
#line 1135
return (a < b) ? a : b; 
#line 1136
} 
#line 1138
static inline int max(int a, int b) 
#line 1139
{ 
#line 1140
return (a > b) ? a : b; 
#line 1141
} 
#line 1143
static inline unsigned umax(unsigned a, unsigned b) 
#line 1144
{ 
#line 1145
return (a > b) ? a : b; 
#line 1146
} 
#line 1148
static inline __int64 llmax(__int64 a, __int64 b) 
#line 1149
{ 
#line 1150
return (a > b) ? a : b; 
#line 1151
} 
#line 1153
static inline unsigned __int64 ullmax(unsigned __int64 a, unsigned __int64 
#line 1154
b) 
#line 1155
{ 
#line 1156
return (a > b) ? a : b; 
#line 1157
} 
#line 1160
#pragma warning(default: 4211)
#line 77 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_surface_types.h"
template< class T, int dim = 1> 
#line 78
struct surface : public surfaceReference { 
#line 81
surface() 
#line 82
{ 
#line 83
(channelDesc) = cudaCreateChannelDesc< T> (); 
#line 84
} 
#line 86
surface(::cudaChannelFormatDesc desc) 
#line 87
{ 
#line 88
(channelDesc) = desc; 
#line 89
} 
#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_surface_types.h"
}; 
#line 93
template< int dim> 
#line 94
struct surface< void, dim>  : public surfaceReference { 
#line 97
surface() 
#line 98
{ 
#line 99
(channelDesc) = cudaCreateChannelDesc< void> (); 
#line 100
} 
#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_surface_types.h"
}; 
#line 77 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
#line 78
struct texture : public textureReference { 
#line 81
texture(int norm = 0, ::cudaTextureFilterMode 
#line 82
fMode = cudaFilterModePoint, ::cudaTextureAddressMode 
#line 83
aMode = cudaAddressModeClamp) 
#line 84
{ 
#line 85
(normalized) = norm; 
#line 86
(filterMode) = fMode; 
#line 87
((addressMode)[0]) = aMode; 
#line 88
((addressMode)[1]) = aMode; 
#line 89
((addressMode)[2]) = aMode; 
#line 90
(channelDesc) = cudaCreateChannelDesc< T> (); 
#line 91
(sRGB) = 0; 
#line 92
} 
#line 94
texture(int norm, ::cudaTextureFilterMode 
#line 95
fMode, ::cudaTextureAddressMode 
#line 96
aMode, ::cudaChannelFormatDesc 
#line 97
desc) 
#line 98
{ 
#line 99
(normalized) = norm; 
#line 100
(filterMode) = fMode; 
#line 101
((addressMode)[0]) = aMode; 
#line 102
((addressMode)[1]) = aMode; 
#line 103
((addressMode)[2]) = aMode; 
#line 104
(channelDesc) = desc; 
#line 105
(sRGB) = 0; 
#line 106
} 
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\cuda_texture_types.h"
}; 
#line 79 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt/device_functions.h"
extern "C" {
#line 3201
}
#line 3209
static __inline int mulhi(int a, int b); 
#line 3211
static __inline unsigned mulhi(unsigned a, unsigned b); 
#line 3213
static __inline unsigned mulhi(int a, unsigned b); 
#line 3215
static __inline unsigned mulhi(unsigned a, int b); 
#line 3217
static __inline __int64 mul64hi(__int64 a, __int64 b); 
#line 3219
static __inline unsigned __int64 mul64hi(unsigned __int64 a, unsigned __int64 b); 
#line 3221
static __inline unsigned __int64 mul64hi(__int64 a, unsigned __int64 b); 
#line 3223
static __inline unsigned __int64 mul64hi(unsigned __int64 a, __int64 b); 
#line 3225
static __inline int float_as_int(float a); 
#line 3227
static __inline float int_as_float(int a); 
#line 3229
static __inline unsigned float_as_uint(float a); 
#line 3231
static __inline float uint_as_float(unsigned a); 
#line 3233
static __inline float saturate(float a); 
#line 3235
static __inline int mul24(int a, int b); 
#line 3237
static __inline unsigned umul24(unsigned a, unsigned b); 
#line 3239
static __inline int float2int(float a, cudaRoundMode mode = cudaRoundZero); 
#line 3241
static __inline unsigned float2uint(float a, cudaRoundMode mode = cudaRoundZero); 
#line 3243
static __inline float int2float(int a, cudaRoundMode mode = cudaRoundNearest); 
#line 3245
static __inline float uint2float(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
#line 80 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline int mulhi(int a, int b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 81
{ 
#line 82
return __mulhi(a, b); 
#line 83
} 
#endif
#line 85 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline unsigned mulhi(unsigned a, unsigned b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 86
{ 
#line 87
return __umulhi(a, b); 
#line 88
} 
#endif
#line 90 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline unsigned mulhi(int a, unsigned b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 91
{ 
#line 92
return __umulhi((unsigned)a, b); 
#line 93
} 
#endif
#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline unsigned mulhi(unsigned a, int b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 96
{ 
#line 97
return __umulhi(a, (unsigned)b); 
#line 98
} 
#endif
#line 100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline __int64 mul64hi(__int64 a, __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 101
{ 
#line 102
return __mul64hi(a, b); 
#line 103
} 
#endif
#line 105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(unsigned __int64 a, unsigned __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 106
{ 
#line 107
return __umul64hi(a, b); 
#line 108
} 
#endif
#line 110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(__int64 a, unsigned __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 111
{ 
#line 112
return __umul64hi((unsigned __int64)a, b); 
#line 113
} 
#endif
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(unsigned __int64 a, __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 116
{ 
#line 117
return __umul64hi(a, (unsigned __int64)b); 
#line 118
} 
#endif
#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline int float_as_int(float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 121
{ 
#line 122
return __float_as_int(a); 
#line 123
} 
#endif
#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline float int_as_float(int a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 126
{ 
#line 127
return __int_as_float(a); 
#line 128
} 
#endif
#line 130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline unsigned float_as_uint(float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 131
{ 
#line 132
return __float_as_uint(a); 
#line 133
} 
#endif
#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline float uint_as_float(unsigned a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 136
{ 
#line 137
return __uint_as_float(a); 
#line 138
} 
#endif
#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline float saturate(float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 140
{ 
#line 141
return __saturatef(a); 
#line 142
} 
#endif
#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline int mul24(int a, int b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 145
{ 
#line 146
return __mul24(a, b); 
#line 147
} 
#endif
#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline unsigned umul24(unsigned a, unsigned b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 150
{ 
#line 151
return __umul24(a, b); 
#line 152
} 
#endif
#line 154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline int float2int(float a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 155
{ 
#line 156
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
#line 160
} 
#endif
#line 162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline unsigned float2uint(float a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 163
{ 
#line 164
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
#line 168
} 
#endif
#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline float int2float(int a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 171
{ 
#line 172
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
#line 176
} 
#endif
#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_functions.hpp"
static __inline float uint2float(unsigned a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 179
{ 
#line 180
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
#line 184
} 
#endif
#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 115
{ } 
#endif
#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 117
{ } 
#endif
#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 121
{ } 
#endif
#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 123
{ } 
#endif
#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 125
{ } 
#endif
#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 127
{ } 
#endif
#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 131
{ } 
#endif
#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 133
{ } 
#endif
#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 135
{ } 
#endif
#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 137
{ } 
#endif
#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 139
{ } 
#endif
#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 141
{ } 
#endif
#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 143
{ } 
#endif
#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 145
{ } 
#endif
#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 147
{ } 
#endif
#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
extern "C" {
#line 182
}
#line 191
static __inline unsigned __int64 atomicAdd(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicExch(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 193
{ } 
#endif
#line 195 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicCAS(unsigned __int64 *address, unsigned __int64 compare, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 195
{ } 
#endif
#line 197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
__declspec(deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning).")) static __inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 197
{ } 
#endif
#line 199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_atomic_functions.h"
__declspec(deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning).")) static __inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 199
{ } 
#endif
#line 77 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.h"
extern "C" {
#line 1129
}
#line 1137
static __inline double fma(double a, double b, double c, cudaRoundMode mode); 
#line 1139
static __inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1141
static __inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1143
static __inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1145
static __inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1147
static __inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1149
static __inline __int64 double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1151
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1153
static __inline double ll2double(__int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1155
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1157
static __inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
#line 1159
static __inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
#line 1161
static __inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
#line 83 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline double fma(double a, double b, double c, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;::exit(___);}
#if 0
#line 84
{ 
#line 85
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
#line 89
} 
#endif
#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline double dmul(double a, double b, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 92
{ 
#line 93
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
#line 97
} 
#endif
#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline double dadd(double a, double b, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 100
{ 
#line 101
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
#line 105
} 
#endif
#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline double dsub(double a, double b, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 108
{ 
#line 109
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
#line 113
} 
#endif
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline int double2int(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 116
{ 
#line 117
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
#line 121
} 
#endif
#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline unsigned double2uint(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 124
{ 
#line 125
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
#line 129
} 
#endif
#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline __int64 double2ll(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 132
{ 
#line 133
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
#line 137
} 
#endif
#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 140
{ 
#line 141
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
#line 145
} 
#endif
#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline double ll2double(__int64 a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 148
{ 
#line 149
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
#line 153
} 
#endif
#line 155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 156
{ 
#line 157
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
#line 161
} 
#endif
#line 163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline double int2double(int a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 164
{ 
#line 165
return (double)a; 
#line 166
} 
#endif
#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline double uint2double(unsigned a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 169
{ 
#line 170
return (double)a; 
#line 171
} 
#endif
#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\device_double_functions.hpp"
static __inline double float2double(float a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 174
{ 
#line 175
return (double)a; 
#line 176
} 
#endif
#line 90 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_20_atomic_functions.h"
static __inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 90
{ } 
#endif
#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMin(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMax(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicAnd(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicOr(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicXor(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMin(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMax(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicAnd(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 115
{ } 
#endif
#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicOr(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 117
{ } 
#endif
#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicXor(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 304
{ } 
#endif
#line 307 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 307
{ } 
#endif
#line 310 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 310
{ } 
#endif
#line 313 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 313
{ } 
#endif
#line 316 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 316
{ } 
#endif
#line 319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 319
{ } 
#endif
#line 322 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 322
{ } 
#endif
#line 325 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 325
{ } 
#endif
#line 328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 328
{ } 
#endif
#line 331 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 331
{ } 
#endif
#line 334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 334
{ } 
#endif
#line 337 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 337
{ } 
#endif
#line 340 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 340
{ } 
#endif
#line 343 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 343
{ } 
#endif
#line 346 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 346
{ } 
#endif
#line 349 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 349
{ } 
#endif
#line 352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 352
{ } 
#endif
#line 355 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 355
{ } 
#endif
#line 358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 358
{ } 
#endif
#line 361 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 361
{ } 
#endif
#line 364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 364
{ } 
#endif
#line 367 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 367
{ } 
#endif
#line 370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 370
{ } 
#endif
#line 373 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 373
{ } 
#endif
#line 376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 376
{ } 
#endif
#line 379 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 379
{ } 
#endif
#line 382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 382
{ } 
#endif
#line 385 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 385
{ } 
#endif
#line 388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 388
{ } 
#endif
#line 391 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 391
{ } 
#endif
#line 394 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 394
{ } 
#endif
#line 397 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 397
{ } 
#endif
#line 400 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 400
{ } 
#endif
#line 403 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 403
{ } 
#endif
#line 406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 406
{ } 
#endif
#line 409 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 409
{ } 
#endif
#line 412 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 412
{ } 
#endif
#line 415 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 415
{ } 
#endif
#line 418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 418
{ } 
#endif
#line 421 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 421
{ } 
#endif
#line 424 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 424
{ } 
#endif
#line 427 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 427
{ } 
#endif
#line 430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 430
{ } 
#endif
#line 433 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 433
{ } 
#endif
#line 436 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 436
{ } 
#endif
#line 439 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
#line 440
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 440
{ } 
#endif
#line 443 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
#line 444
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 444
{ } 
#endif
#line 447 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_block(unsigned __int64 *address, unsigned __int64 
#line 448
compare, unsigned __int64 
#line 449
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 449
{ } 
#endif
#line 452 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_system(unsigned __int64 *address, unsigned __int64 
#line 453
compare, unsigned __int64 
#line 454
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 454
{ } 
#endif
#line 457 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 457
{ } 
#endif
#line 460 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 460
{ } 
#endif
#line 463 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 463
{ } 
#endif
#line 466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 466
{ } 
#endif
#line 469 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 469
{ } 
#endif
#line 472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 472
{ } 
#endif
#line 475 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 475
{ } 
#endif
#line 478 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 478
{ } 
#endif
#line 481 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 481
{ } 
#endif
#line 484 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 484
{ } 
#endif
#line 487 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 487
{ } 
#endif
#line 490 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 490
{ } 
#endif
#line 493 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 493
{ } 
#endif
#line 496 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 496
{ } 
#endif
#line 499 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 499
{ } 
#endif
#line 502 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 502
{ } 
#endif
#line 505 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 505
{ } 
#endif
#line 508 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 508
{ } 
#endif
#line 511 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 511
{ } 
#endif
#line 514 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 514
{ } 
#endif
#line 517 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 517
{ } 
#endif
#line 520 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 520
{ } 
#endif
#line 523 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 523
{ } 
#endif
#line 526 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 526
{ } 
#endif
#line 92 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_20_intrinsics.h"
extern "C" {
#line 1477
}
#line 1484
__declspec(deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning).")) static __inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1484
{ } 
#endif
#line 1486 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_20_intrinsics.h"
static __inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1486
{ } 
#endif
#line 1488 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1488
{ } 
#endif
#line 1490 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1490
{ } 
#endif
#line 1497 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_20_intrinsics.h"
static __inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1497
{ } 
#endif
#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 114
{ } 
#endif
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
#line 115
{ } 
#endif
#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 123
{ } 
#endif
#line 124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 124
{ } 
#endif
#line 126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 127
{ } 
#endif
#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 132
{ } 
#endif
#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 133
{ } 
#endif
#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 135
{ } 
#endif
#line 136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 138
{ } 
#endif
#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 139
{ } 
#endif
#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 141
{ } 
#endif
#line 142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 145
{ } 
#endif
#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 147
{ } 
#endif
#line 148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 148
{ } 
#endif
#line 150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 150
{ } 
#endif
#line 151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 151
{ } 
#endif
#line 153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 153
{ } 
#endif
#line 154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 154
{ } 
#endif
#line 156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 156
{ } 
#endif
#line 157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 157
{ } 
#endif
#line 160 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline __int64 __shfl(__int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 160
{ } 
#endif
#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_sync(unsigned mask, __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 161
{ } 
#endif
#line 163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned __int64 __shfl(unsigned __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 163
{ } 
#endif
#line 164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_sync(unsigned mask, unsigned __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 164
{ } 
#endif
#line 166 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline __int64 __shfl_up(__int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 166
{ } 
#endif
#line 167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_up_sync(unsigned mask, __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 167
{ } 
#endif
#line 169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned __int64 __shfl_up(unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 169
{ } 
#endif
#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_up_sync(unsigned mask, unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 170
{ } 
#endif
#line 172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline __int64 __shfl_down(__int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 172
{ } 
#endif
#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_down_sync(unsigned mask, __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 173
{ } 
#endif
#line 175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned __int64 __shfl_down(unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 175
{ } 
#endif
#line 176 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_down_sync(unsigned mask, unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 176
{ } 
#endif
#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline __int64 __shfl_xor(__int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 178
{ } 
#endif
#line 179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_xor_sync(unsigned mask, __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 179
{ } 
#endif
#line 181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned __int64 __shfl_xor(unsigned __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 181
{ } 
#endif
#line 182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_xor_sync(unsigned mask, unsigned __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 182
{ } 
#endif
#line 184 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 184
{ } 
#endif
#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 185
{ } 
#endif
#line 187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 187
{ } 
#endif
#line 188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 188
{ } 
#endif
#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 190
{ } 
#endif
#line 191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 193
{ } 
#endif
#line 194 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 194
{ } 
#endif
#line 198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 198
{ } 
#endif
#line 199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 199
{ } 
#endif
#line 201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 201
{ } 
#endif
#line 202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 202
{ } 
#endif
#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 204
{ } 
#endif
#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 205
{ } 
#endif
#line 207 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 207
{ } 
#endif
#line 208 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 208
{ } 
#endif
#line 210 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 210
{ } 
#endif
#line 211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 211
{ } 
#endif
#line 213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 213
{ } 
#endif
#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 214
{ } 
#endif
#line 216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 216
{ } 
#endif
#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 217
{ } 
#endif
#line 219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 219
{ } 
#endif
#line 220 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 220
{ } 
#endif
#line 89 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 89
{ } 
#endif
#line 90 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 90
{ } 
#endif
#line 92 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldg(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldg(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 114
{ } 
#endif
#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 115
{ } 
#endif
#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 117
{ } 
#endif
#line 118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 118
{ } 
#endif
#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 120
{ } 
#endif
#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 121
{ } 
#endif
#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 125
{ } 
#endif
#line 126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 128
{ } 
#endif
#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 131
{ } 
#endif
#line 132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcg(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 132
{ } 
#endif
#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 133
{ } 
#endif
#line 134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 134
{ } 
#endif
#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 135
{ } 
#endif
#line 136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 137
{ } 
#endif
#line 138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 138
{ } 
#endif
#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 139
{ } 
#endif
#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 141
{ } 
#endif
#line 142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 143
{ } 
#endif
#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcg(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 145
{ } 
#endif
#line 146 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 146
{ } 
#endif
#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 147
{ } 
#endif
#line 148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 148
{ } 
#endif
#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 149
{ } 
#endif
#line 150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 150
{ } 
#endif
#line 151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 151
{ } 
#endif
#line 153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 153
{ } 
#endif
#line 154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 154
{ } 
#endif
#line 155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 155
{ } 
#endif
#line 156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 156
{ } 
#endif
#line 157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 157
{ } 
#endif
#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 161
{ } 
#endif
#line 162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 162
{ } 
#endif
#line 164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 164
{ } 
#endif
#line 165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 165
{ } 
#endif
#line 166 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 166
{ } 
#endif
#line 167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 167
{ } 
#endif
#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldca(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 168
{ } 
#endif
#line 169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 169
{ } 
#endif
#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 170
{ } 
#endif
#line 171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 171
{ } 
#endif
#line 172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 172
{ } 
#endif
#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 173
{ } 
#endif
#line 174 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 174
{ } 
#endif
#line 175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 175
{ } 
#endif
#line 177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 177
{ } 
#endif
#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 178
{ } 
#endif
#line 179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 179
{ } 
#endif
#line 180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldca(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 180
{ } 
#endif
#line 181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 181
{ } 
#endif
#line 182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 182
{ } 
#endif
#line 183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 183
{ } 
#endif
#line 184 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 184
{ } 
#endif
#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 185
{ } 
#endif
#line 186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 186
{ } 
#endif
#line 187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 187
{ } 
#endif
#line 189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 189
{ } 
#endif
#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 190
{ } 
#endif
#line 191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 192
{ } 
#endif
#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 193
{ } 
#endif
#line 197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 197
{ } 
#endif
#line 198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 198
{ } 
#endif
#line 200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 200
{ } 
#endif
#line 201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 201
{ } 
#endif
#line 202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 202
{ } 
#endif
#line 203 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 203
{ } 
#endif
#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcs(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 204
{ } 
#endif
#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 205
{ } 
#endif
#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 206
{ } 
#endif
#line 207 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 207
{ } 
#endif
#line 208 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 208
{ } 
#endif
#line 209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 209
{ } 
#endif
#line 210 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 210
{ } 
#endif
#line 211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 211
{ } 
#endif
#line 213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 213
{ } 
#endif
#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 214
{ } 
#endif
#line 215 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 215
{ } 
#endif
#line 216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcs(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 216
{ } 
#endif
#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 217
{ } 
#endif
#line 218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 218
{ } 
#endif
#line 219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 219
{ } 
#endif
#line 220 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 220
{ } 
#endif
#line 221 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 221
{ } 
#endif
#line 222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 222
{ } 
#endif
#line 223 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 223
{ } 
#endif
#line 225 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 225
{ } 
#endif
#line 226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 226
{ } 
#endif
#line 227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 227
{ } 
#endif
#line 228 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 228
{ } 
#endif
#line 229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 229
{ } 
#endif
#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 246
{ } 
#endif
#line 258 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 258
{ } 
#endif
#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 271
{ } 
#endif
#line 283 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 283
{ } 
#endif
#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 91
{ } 
#endif
#line 92 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 83 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 83
{ } 
#endif
#line 84 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 84
{ } 
#endif
#line 85 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 85
{ } 
#endif
#line 86 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 86
{ } 
#endif
#line 87 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned __int64 value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 87
{ } 
#endif
#line 88 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, __int64 value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 88
{ } 
#endif
#line 89 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 89
{ } 
#endif
#line 90 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 90
{ } 
#endif
#line 92 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned __int64 value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, __int64 value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt\\sm_70_rt.h"
static __inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 116 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 117
surf1Dread(T *res, ::surface< void, 1>  surf, int x, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;::exit(___);}
#if 0
#line 118
{ 
#line 122
} 
#endif
#line 124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 125
surf1Dread(::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)mode;::exit(___);}
#if 0
#line 126
{ 
#line 132
} 
#endif
#line 134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 135
surf1Dread(T *res, ::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;::exit(___);}
#if 0
#line 136
{ 
#line 140
} 
#endif
#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 144
surf2Dread(T *res, ::surface< void, 2>  surf, int x, int y, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;::exit(___);}
#if 0
#line 145
{ 
#line 149
} 
#endif
#line 151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 152
surf2Dread(::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 153
{ 
#line 159
} 
#endif
#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 162
surf2Dread(T *res, ::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 163
{ 
#line 167
} 
#endif
#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 171
surf3Dread(T *res, ::surface< void, 3>  surf, int x, int y, int z, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;::exit(___);}
#if 0
#line 172
{ 
#line 176
} 
#endif
#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 179
surf3Dread(::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 180
{ 
#line 186
} 
#endif
#line 188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 189
surf3Dread(T *res, ::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 190
{ 
#line 194
} 
#endif
#line 198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 199
surf1DLayeredread(T *res, ::surface< void, 241>  surf, int x, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 200
{ 
#line 204
} 
#endif
#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 207
surf1DLayeredread(::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 208
{ 
#line 214
} 
#endif
#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 218
surf1DLayeredread(T *res, ::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 219
{ 
#line 223
} 
#endif
#line 226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 227
surf2DLayeredread(T *res, ::surface< void, 242>  surf, int x, int y, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 228
{ 
#line 232
} 
#endif
#line 234 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 235
surf2DLayeredread(::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 236
{ 
#line 242
} 
#endif
#line 245 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 246
surf2DLayeredread(T *res, ::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 247
{ 
#line 251
} 
#endif
#line 254 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 255
surfCubemapread(T *res, ::surface< void, 12>  surf, int x, int y, int face, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;::exit(___);}
#if 0
#line 256
{ 
#line 260
} 
#endif
#line 262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 263
surfCubemapread(::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 264
{ 
#line 271
} 
#endif
#line 273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 274
surfCubemapread(T *res, ::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 275
{ 
#line 279
} 
#endif
#line 282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 283
surfCubemapLayeredread(T *res, ::surface< void, 252>  surf, int x, int y, int layerFace, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;::exit(___);}
#if 0
#line 284
{ 
#line 288
} 
#endif
#line 290 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 291
surfCubemapLayeredread(::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;::exit(___);}
#if 0
#line 292
{ 
#line 298
} 
#endif
#line 300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 301
surfCubemapLayeredread(T *res, ::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;::exit(___);}
#if 0
#line 302
{ 
#line 306
} 
#endif
#line 309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 310
surf1Dwrite(T val, ::surface< void, 1>  surf, int x, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;::exit(___);}
#if 0
#line 311
{ 
#line 315
} 
#endif
#line 317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 318
surf1Dwrite(T val, ::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;::exit(___);}
#if 0
#line 319
{ 
#line 323
} 
#endif
#line 327 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 328
surf2Dwrite(T val, ::surface< void, 2>  surf, int x, int y, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;::exit(___);}
#if 0
#line 329
{ 
#line 333
} 
#endif
#line 335 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 336
surf2Dwrite(T val, ::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 337
{ 
#line 341
} 
#endif
#line 344 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 345
surf3Dwrite(T val, ::surface< void, 3>  surf, int x, int y, int z, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;::exit(___);}
#if 0
#line 346
{ 
#line 350
} 
#endif
#line 352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 353
surf3Dwrite(T val, ::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 354
{ 
#line 358
} 
#endif
#line 361 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 362
surf1DLayeredwrite(T val, ::surface< void, 241>  surf, int x, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 363
{ 
#line 367
} 
#endif
#line 369 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 370
surf1DLayeredwrite(T val, ::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 371
{ 
#line 375
} 
#endif
#line 378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 379
surf2DLayeredwrite(T val, ::surface< void, 242>  surf, int x, int y, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 380
{ 
#line 384
} 
#endif
#line 386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 387
surf2DLayeredwrite(T val, ::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 388
{ 
#line 392
} 
#endif
#line 395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 396
surfCubemapwrite(T val, ::surface< void, 12>  surf, int x, int y, int face, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;::exit(___);}
#if 0
#line 397
{ 
#line 401
} 
#endif
#line 403 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 404
surfCubemapwrite(T val, ::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 405
{ 
#line 409
} 
#endif
#line 413 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 414
surfCubemapLayeredwrite(T val, ::surface< void, 252>  surf, int x, int y, int layerFace, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;::exit(___);}
#if 0
#line 415
{ 
#line 419
} 
#endif
#line 421 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 422
surfCubemapLayeredwrite(T val, ::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;::exit(___);}
#if 0
#line 423
{ 
#line 427
} 
#endif
#line 68 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> 
#line 69
struct __nv_tex_rmet_ret { }; 
#line 71
template<> struct __nv_tex_rmet_ret< char>  { typedef char type; }; 
#line 72
template<> struct __nv_tex_rmet_ret< signed char>  { typedef signed char type; }; 
#line 73
template<> struct __nv_tex_rmet_ret< unsigned char>  { typedef unsigned char type; }; 
#line 74
template<> struct __nv_tex_rmet_ret< char1>  { typedef char1 type; }; 
#line 75
template<> struct __nv_tex_rmet_ret< uchar1>  { typedef uchar1 type; }; 
#line 76
template<> struct __nv_tex_rmet_ret< char2>  { typedef char2 type; }; 
#line 77
template<> struct __nv_tex_rmet_ret< uchar2>  { typedef uchar2 type; }; 
#line 78
template<> struct __nv_tex_rmet_ret< char4>  { typedef char4 type; }; 
#line 79
template<> struct __nv_tex_rmet_ret< uchar4>  { typedef uchar4 type; }; 
#line 81
template<> struct __nv_tex_rmet_ret< short>  { typedef short type; }; 
#line 82
template<> struct __nv_tex_rmet_ret< unsigned short>  { typedef unsigned short type; }; 
#line 83
template<> struct __nv_tex_rmet_ret< short1>  { typedef short1 type; }; 
#line 84
template<> struct __nv_tex_rmet_ret< ushort1>  { typedef ushort1 type; }; 
#line 85
template<> struct __nv_tex_rmet_ret< short2>  { typedef short2 type; }; 
#line 86
template<> struct __nv_tex_rmet_ret< ushort2>  { typedef ushort2 type; }; 
#line 87
template<> struct __nv_tex_rmet_ret< short4>  { typedef short4 type; }; 
#line 88
template<> struct __nv_tex_rmet_ret< ushort4>  { typedef ushort4 type; }; 
#line 90
template<> struct __nv_tex_rmet_ret< int>  { typedef int type; }; 
#line 91
template<> struct __nv_tex_rmet_ret< unsigned>  { typedef unsigned type; }; 
#line 92
template<> struct __nv_tex_rmet_ret< int1>  { typedef int1 type; }; 
#line 93
template<> struct __nv_tex_rmet_ret< uint1>  { typedef uint1 type; }; 
#line 94
template<> struct __nv_tex_rmet_ret< int2>  { typedef int2 type; }; 
#line 95
template<> struct __nv_tex_rmet_ret< uint2>  { typedef uint2 type; }; 
#line 96
template<> struct __nv_tex_rmet_ret< int4>  { typedef int4 type; }; 
#line 97
template<> struct __nv_tex_rmet_ret< uint4>  { typedef uint4 type; }; 
#line 100
template<> struct __nv_tex_rmet_ret< long>  { typedef long type; }; 
#line 101
template<> struct __nv_tex_rmet_ret< unsigned long>  { typedef unsigned long type; }; 
#line 102
template<> struct __nv_tex_rmet_ret< long1>  { typedef long1 type; }; 
#line 103
template<> struct __nv_tex_rmet_ret< ulong1>  { typedef ulong1 type; }; 
#line 104
template<> struct __nv_tex_rmet_ret< long2>  { typedef long2 type; }; 
#line 105
template<> struct __nv_tex_rmet_ret< ulong2>  { typedef ulong2 type; }; 
#line 106
template<> struct __nv_tex_rmet_ret< long4>  { typedef long4 type; }; 
#line 107
template<> struct __nv_tex_rmet_ret< ulong4>  { typedef ulong4 type; }; 
#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret< float>  { typedef float type; }; 
#line 110
template<> struct __nv_tex_rmet_ret< float1>  { typedef float1 type; }; 
#line 111
template<> struct __nv_tex_rmet_ret< float2>  { typedef float2 type; }; 
#line 112
template<> struct __nv_tex_rmet_ret< float4>  { typedef float4 type; }; 
#line 115
template< class T> struct __nv_tex_rmet_cast { typedef T *type; }; 
#line 117
template<> struct __nv_tex_rmet_cast< long>  { typedef int *type; }; 
#line 118
template<> struct __nv_tex_rmet_cast< unsigned long>  { typedef unsigned *type; }; 
#line 119
template<> struct __nv_tex_rmet_cast< long1>  { typedef int1 *type; }; 
#line 120
template<> struct __nv_tex_rmet_cast< ulong1>  { typedef uint1 *type; }; 
#line 121
template<> struct __nv_tex_rmet_cast< long2>  { typedef int2 *type; }; 
#line 122
template<> struct __nv_tex_rmet_cast< ulong2>  { typedef uint2 *type; }; 
#line 123
template<> struct __nv_tex_rmet_cast< long4>  { typedef int4 *type; }; 
#line 124
template<> struct __nv_tex_rmet_cast< ulong4>  { typedef uint4 *type; }; 
#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 128
tex1Dfetch(texture< T, 1, cudaReadModeElementType>  t, int x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 129
{ 
#line 135
} 
#endif
#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> 
#line 138
struct __nv_tex_rmnf_ret { }; 
#line 140
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
#line 141
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
#line 142
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
#line 143
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
#line 144
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
#line 145
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
#line 146
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
#line 147
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
#line 148
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
#line 149
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
#line 150
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
#line 151
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
#line 152
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
#line 153
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
#line 154
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
#line 155
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
#line 156
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
#line 158
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 159
tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat>  t, int x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 160
{ 
#line 167
} 
#endif
#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 171
tex1D(texture< T, 1, cudaReadModeElementType>  t, float x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 172
{ 
#line 178
} 
#endif
#line 180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 181
tex1D(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 182
{ 
#line 189
} 
#endif
#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 194
tex2D(texture< T, 2, cudaReadModeElementType>  t, float x, float y) {int volatile ___ = 1;(void)t;(void)x;(void)y;::exit(___);}
#if 0
#line 195
{ 
#line 202
} 
#endif
#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 205
tex2D(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y) {int volatile ___ = 1;(void)t;(void)x;(void)y;::exit(___);}
#if 0
#line 206
{ 
#line 213
} 
#endif
#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 218
tex1DLayered(texture< T, 241, cudaReadModeElementType>  t, float x, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)layer;::exit(___);}
#if 0
#line 219
{ 
#line 225
} 
#endif
#line 227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 228
tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)layer;::exit(___);}
#if 0
#line 229
{ 
#line 236
} 
#endif
#line 240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 241
tex2DLayered(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 242
{ 
#line 248
} 
#endif
#line 250 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 251
tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 252
{ 
#line 259
} 
#endif
#line 262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 263
tex3D(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 264
{ 
#line 270
} 
#endif
#line 272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 273
tex3D(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 274
{ 
#line 281
} 
#endif
#line 284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 285
texCubemap(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 286
{ 
#line 292
} 
#endif
#line 294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 295
texCubemap(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 296
{ 
#line 303
} 
#endif
#line 306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> 
#line 307
struct __nv_tex2dgather_ret { }; 
#line 308
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
#line 309
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
#line 310
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
#line 311
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
#line 312
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
#line 313
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
#line 314
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
#line 315
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
#line 316
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
#line 317
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
#line 318
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
#line 320
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
#line 321
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
#line 322
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
#line 323
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
#line 324
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
#line 325
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
#line 326
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
#line 327
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
#line 328
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
#line 329
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
#line 331
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
#line 332
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
#line 333
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
#line 334
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
#line 335
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
#line 336
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
#line 337
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
#line 338
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
#line 339
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
#line 340
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
#line 342
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
#line 343
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
#line 344
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
#line 345
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
#line 346
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
#line 348
template< class T> static __forceinline typename __nv_tex2dgather_ret< T> ::type 
#line 349
tex2Dgather(texture< T, 2, cudaReadModeElementType>  t, float x, float y, int comp = 0) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 350
{ 
#line 357
} 
#endif
#line 360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> struct __nv_tex2dgather_rmnf_ret { }; 
#line 361
template<> struct __nv_tex2dgather_rmnf_ret< char>  { typedef float4 type; }; 
#line 362
template<> struct __nv_tex2dgather_rmnf_ret< signed char>  { typedef float4 type; }; 
#line 363
template<> struct __nv_tex2dgather_rmnf_ret< unsigned char>  { typedef float4 type; }; 
#line 364
template<> struct __nv_tex2dgather_rmnf_ret< char1>  { typedef float4 type; }; 
#line 365
template<> struct __nv_tex2dgather_rmnf_ret< uchar1>  { typedef float4 type; }; 
#line 366
template<> struct __nv_tex2dgather_rmnf_ret< char2>  { typedef float4 type; }; 
#line 367
template<> struct __nv_tex2dgather_rmnf_ret< uchar2>  { typedef float4 type; }; 
#line 368
template<> struct __nv_tex2dgather_rmnf_ret< char3>  { typedef float4 type; }; 
#line 369
template<> struct __nv_tex2dgather_rmnf_ret< uchar3>  { typedef float4 type; }; 
#line 370
template<> struct __nv_tex2dgather_rmnf_ret< char4>  { typedef float4 type; }; 
#line 371
template<> struct __nv_tex2dgather_rmnf_ret< uchar4>  { typedef float4 type; }; 
#line 372
template<> struct __nv_tex2dgather_rmnf_ret< signed short>  { typedef float4 type; }; 
#line 373
template<> struct __nv_tex2dgather_rmnf_ret< unsigned short>  { typedef float4 type; }; 
#line 374
template<> struct __nv_tex2dgather_rmnf_ret< short1>  { typedef float4 type; }; 
#line 375
template<> struct __nv_tex2dgather_rmnf_ret< ushort1>  { typedef float4 type; }; 
#line 376
template<> struct __nv_tex2dgather_rmnf_ret< short2>  { typedef float4 type; }; 
#line 377
template<> struct __nv_tex2dgather_rmnf_ret< ushort2>  { typedef float4 type; }; 
#line 378
template<> struct __nv_tex2dgather_rmnf_ret< short3>  { typedef float4 type; }; 
#line 379
template<> struct __nv_tex2dgather_rmnf_ret< ushort3>  { typedef float4 type; }; 
#line 380
template<> struct __nv_tex2dgather_rmnf_ret< short4>  { typedef float4 type; }; 
#line 381
template<> struct __nv_tex2dgather_rmnf_ret< ushort4>  { typedef float4 type; }; 
#line 383
template< class T> static __forceinline typename __nv_tex2dgather_rmnf_ret< T> ::type 
#line 384
tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, int comp = 0) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 385
{ 
#line 392
} 
#endif
#line 396 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 397
tex1DLod(texture< T, 1, cudaReadModeElementType>  t, float x, float level) {int volatile ___ = 1;(void)t;(void)x;(void)level;::exit(___);}
#if 0
#line 398
{ 
#line 404
} 
#endif
#line 406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 407
tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float level) {int volatile ___ = 1;(void)t;(void)x;(void)level;::exit(___);}
#if 0
#line 408
{ 
#line 415
} 
#endif
#line 418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 419
tex2DLod(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 420
{ 
#line 426
} 
#endif
#line 428 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 429
tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 430
{ 
#line 437
} 
#endif
#line 440 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 441
tex1DLayeredLod(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 442
{ 
#line 448
} 
#endif
#line 450 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 451
tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 452
{ 
#line 459
} 
#endif
#line 462 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 463
tex2DLayeredLod(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 464
{ 
#line 470
} 
#endif
#line 472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 473
tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 474
{ 
#line 481
} 
#endif
#line 484 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 485
tex3DLod(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 486
{ 
#line 492
} 
#endif
#line 494 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 495
tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 496
{ 
#line 503
} 
#endif
#line 506 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 507
texCubemapLod(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 508
{ 
#line 514
} 
#endif
#line 516 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 517
texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 518
{ 
#line 525
} 
#endif
#line 529 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 530
texCubemapLayered(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 531
{ 
#line 537
} 
#endif
#line 539 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 540
texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 541
{ 
#line 548
} 
#endif
#line 552 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 553
texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 554
{ 
#line 560
} 
#endif
#line 562 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 563
texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 564
{ 
#line 571
} 
#endif
#line 575 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 576
texCubemapGrad(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 577
{ 
#line 583
} 
#endif
#line 585 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 586
texCubemapGrad(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 587
{ 
#line 594
} 
#endif
#line 598 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 599
texCubemapLayeredGrad(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 600
{ 
#line 606
} 
#endif
#line 608 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 609
texCubemapLayeredGrad(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 610
{ 
#line 617
} 
#endif
#line 621 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 622
tex1DGrad(texture< T, 1, cudaReadModeElementType>  t, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 623
{ 
#line 629
} 
#endif
#line 631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 632
tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 633
{ 
#line 640
} 
#endif
#line 644 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 645
tex2DGrad(texture< T, 2, cudaReadModeElementType>  t, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 646
{ 
#line 652
} 
#endif
#line 654 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 655
tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 656
{ 
#line 663
} 
#endif
#line 666 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 667
tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 668
{ 
#line 674
} 
#endif
#line 676 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 677
tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 678
{ 
#line 685
} 
#endif
#line 688 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 689
tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 690
{ 
#line 696
} 
#endif
#line 698 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 699
tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 700
{ 
#line 707
} 
#endif
#line 710 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 711
tex3DGrad(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 712
{ 
#line 718
} 
#endif
#line 720 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 721
tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 722
{ 
#line 729
} 
#endif
#line 61 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
#line 62
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
#line 63
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
#line 64
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
#line 65
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
#line 66
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
#line 67
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
#line 68
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
#line 69
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
#line 70
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
#line 71
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
#line 72
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
#line 73
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
#line 74
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
#line 75
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
#line 76
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
#line 77
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
#line 78
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
#line 79
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
#line 80
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
#line 81
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
#line 82
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
#line 83
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
#line 84
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
#line 85
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
#line 86
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
#line 88
template<> struct __nv_itex_trait< long>  { typedef void type; }; 
#line 89
template<> struct __nv_itex_trait< long1>  { typedef void type; }; 
#line 90
template<> struct __nv_itex_trait< long2>  { typedef void type; }; 
#line 91
template<> struct __nv_itex_trait< long4>  { typedef void type; }; 
#line 92
template<> struct __nv_itex_trait< unsigned long>  { typedef void type; }; 
#line 93
template<> struct __nv_itex_trait< ulong1>  { typedef void type; }; 
#line 94
template<> struct __nv_itex_trait< ulong2>  { typedef void type; }; 
#line 95
template<> struct __nv_itex_trait< ulong4>  { typedef void type; }; 
#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
#line 98
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
#line 99
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
#line 100
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
#line 104
template< class T> static typename __nv_itex_trait< T> ::type 
#line 105
tex1Dfetch(T *ptr, ::cudaTextureObject_t obj, int x) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;::exit(___);}
#if 0
#line 106
{ 
#line 110
} 
#endif
#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 113
tex1Dfetch(::cudaTextureObject_t texObject, int x) {int volatile ___ = 1;(void)texObject;(void)x;::exit(___);}
#if 0
#line 114
{ 
#line 120
} 
#endif
#line 122 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 123
tex1D(T *ptr, ::cudaTextureObject_t obj, float x) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;::exit(___);}
#if 0
#line 124
{ 
#line 128
} 
#endif
#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 132
tex1D(::cudaTextureObject_t texObject, float x) {int volatile ___ = 1;(void)texObject;(void)x;::exit(___);}
#if 0
#line 133
{ 
#line 139
} 
#endif
#line 142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 143
tex2D(T *ptr, ::cudaTextureObject_t obj, float x, float y) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;::exit(___);}
#if 0
#line 144
{ 
#line 148
} 
#endif
#line 150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 151
tex2D(::cudaTextureObject_t texObject, float x, float y) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;::exit(___);}
#if 0
#line 152
{ 
#line 158
} 
#endif
#line 160 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 161
tex3D(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 162
{ 
#line 166
} 
#endif
#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 169
tex3D(::cudaTextureObject_t texObject, float x, float y, float z) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 170
{ 
#line 176
} 
#endif
#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 179
tex1DLayered(T *ptr, ::cudaTextureObject_t obj, float x, int layer) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;::exit(___);}
#if 0
#line 180
{ 
#line 184
} 
#endif
#line 186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 187
tex1DLayered(::cudaTextureObject_t texObject, float x, int layer) {int volatile ___ = 1;(void)texObject;(void)x;(void)layer;::exit(___);}
#if 0
#line 188
{ 
#line 194
} 
#endif
#line 196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 197
tex2DLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 198
{ 
#line 202
} 
#endif
#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 205
tex2DLayered(::cudaTextureObject_t texObject, float x, float y, int layer) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 206
{ 
#line 212
} 
#endif
#line 215 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 216
texCubemap(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 217
{ 
#line 221
} 
#endif
#line 224 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 225
texCubemap(::cudaTextureObject_t texObject, float x, float y, float z) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 226
{ 
#line 232
} 
#endif
#line 235 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 236
texCubemapLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 237
{ 
#line 241
} 
#endif
#line 243 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 244
texCubemapLayered(::cudaTextureObject_t texObject, float x, float y, float z, int layer) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 245
{ 
#line 251
} 
#endif
#line 253 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 254
tex2Dgather(T *ptr, ::cudaTextureObject_t obj, float x, float y, int comp = 0) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 255
{ 
#line 259
} 
#endif
#line 261 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 262
tex2Dgather(::cudaTextureObject_t to, float x, float y, int comp = 0) {int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 263
{ 
#line 269
} 
#endif
#line 273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 274
tex1DLod(T *ptr, ::cudaTextureObject_t obj, float x, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;::exit(___);}
#if 0
#line 275
{ 
#line 279
} 
#endif
#line 281 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 282
tex1DLod(::cudaTextureObject_t texObject, float x, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)level;::exit(___);}
#if 0
#line 283
{ 
#line 289
} 
#endif
#line 292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 293
tex2DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 294
{ 
#line 298
} 
#endif
#line 300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 301
tex2DLod(::cudaTextureObject_t texObject, float x, float y, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 302
{ 
#line 308
} 
#endif
#line 311 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 312
tex3DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 313
{ 
#line 317
} 
#endif
#line 319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 320
tex3DLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 321
{ 
#line 327
} 
#endif
#line 330 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 331
tex1DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, int layer, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 332
{ 
#line 336
} 
#endif
#line 338 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 339
tex1DLayeredLod(::cudaTextureObject_t texObject, float x, int layer, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 340
{ 
#line 346
} 
#endif
#line 349 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 350
tex2DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 351
{ 
#line 355
} 
#endif
#line 357 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 358
tex2DLayeredLod(::cudaTextureObject_t texObject, float x, float y, int layer, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 359
{ 
#line 365
} 
#endif
#line 368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 369
texCubemapLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 370
{ 
#line 374
} 
#endif
#line 376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 377
texCubemapLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 378
{ 
#line 384
} 
#endif
#line 387 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 388
texCubemapGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 389
{ 
#line 393
} 
#endif
#line 395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 396
texCubemapGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 397
{ 
#line 403
} 
#endif
#line 405 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 406
texCubemapLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 407
{ 
#line 411
} 
#endif
#line 413 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 414
texCubemapLayeredLod(::cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 415
{ 
#line 421
} 
#endif
#line 423 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 424
tex1DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 425
{ 
#line 429
} 
#endif
#line 431 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 432
tex1DGrad(::cudaTextureObject_t texObject, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 433
{ 
#line 439
} 
#endif
#line 442 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 443
tex2DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 444
{ 
#line 449
} 
#endif
#line 451 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 452
tex2DGrad(::cudaTextureObject_t texObject, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 453
{ 
#line 459
} 
#endif
#line 462 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 463
tex3DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 464
{ 
#line 468
} 
#endif
#line 470 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 471
tex3DGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 472
{ 
#line 478
} 
#endif
#line 481 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 482
tex1DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 483
{ 
#line 487
} 
#endif
#line 489 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 490
tex1DLayeredGrad(::cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 491
{ 
#line 497
} 
#endif
#line 500 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 501
tex2DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 502
{ 
#line 506
} 
#endif
#line 508 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 509
tex2DLayeredGrad(::cudaTextureObject_t texObject, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 510
{ 
#line 516
} 
#endif
#line 519 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 520
texCubemapLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 521
{ 
#line 525
} 
#endif
#line 527 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 528
texCubemapLayeredGrad(::cudaTextureObject_t texObject, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 529
{ 
#line 535
} 
#endif
#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
#line 61
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
#line 62
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
#line 63
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
#line 64
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
#line 65
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
#line 66
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
#line 67
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
#line 68
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
#line 69
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
#line 70
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
#line 71
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
#line 72
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
#line 73
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
#line 74
template<> struct __nv_isurf_trait< __int64>  { typedef void type; }; 
#line 75
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
#line 76
template<> struct __nv_isurf_trait< unsigned __int64>  { typedef void type; }; 
#line 77
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
#line 78
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
#line 79
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
#line 81
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
#line 82
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
#line 83
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
#line 84
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
#line 85
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
#line 86
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
#line 87
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
#line 88
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
#line 89
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
#line 91
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
#line 92
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
#line 93
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
#line 94
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
#line 95
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
#line 96
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
#line 97
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
#line 100
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 101
surf1Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;::exit(___);}
#if 0
#line 102
{ 
#line 106
} 
#endif
#line 108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 109
surf1Dread(::cudaSurfaceObject_t surfObject, int x, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;::exit(___);}
#if 0
#line 110
{ 
#line 116
} 
#endif
#line 118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 119
surf2Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 120
{ 
#line 124
} 
#endif
#line 126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 127
surf2Dread(::cudaSurfaceObject_t surfObject, int x, int y, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;::exit(___);}
#if 0
#line 128
{ 
#line 134
} 
#endif
#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 138
surf3Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 139
{ 
#line 143
} 
#endif
#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 146
surf3Dread(::cudaSurfaceObject_t surfObject, int x, int y, int z, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;::exit(___);}
#if 0
#line 147
{ 
#line 153
} 
#endif
#line 155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 156
surf1DLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 157
{ 
#line 161
} 
#endif
#line 163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 164
surf1DLayeredread(::cudaSurfaceObject_t surfObject, int x, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;::exit(___);}
#if 0
#line 165
{ 
#line 171
} 
#endif
#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 174
surf2DLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 175
{ 
#line 179
} 
#endif
#line 181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 182
surf2DLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;::exit(___);}
#if 0
#line 183
{ 
#line 189
} 
#endif
#line 191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 192
surfCubemapread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 193
{ 
#line 197
} 
#endif
#line 199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 200
surfCubemapread(::cudaSurfaceObject_t surfObject, int x, int y, int face, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;::exit(___);}
#if 0
#line 201
{ 
#line 207
} 
#endif
#line 209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 210
surfCubemapLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int layerface, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;::exit(___);}
#if 0
#line 211
{ 
#line 215
} 
#endif
#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 218
surfCubemapLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layerface, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;::exit(___);}
#if 0
#line 219
{ 
#line 225
} 
#endif
#line 227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 228
surf1Dwrite(T val, ::cudaSurfaceObject_t obj, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;::exit(___);}
#if 0
#line 229
{ 
#line 233
} 
#endif
#line 235 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 236
surf2Dwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 237
{ 
#line 241
} 
#endif
#line 243 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 244
surf3Dwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 245
{ 
#line 249
} 
#endif
#line 251 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 252
surf1DLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 253
{ 
#line 257
} 
#endif
#line 259 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 260
surf2DLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 261
{ 
#line 265
} 
#endif
#line 267 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 268
surfCubemapwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 269
{ 
#line 273
} 
#endif
#line 275 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 276
surfCubemapLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int layerface, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;::exit(___);}
#if 0
#line 277
{ 
#line 281
} 
#endif
#line 3280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\crt/device_functions.h"
extern "C" unsigned __stdcall __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void * stream = 0); 
#line 68 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_launch_parameters.h"
extern "C" {
#line 71 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v9.2\\include\\device_launch_parameters.h"
extern const uint3 __device_builtin_variable_threadIdx; 
#line 72
extern const uint3 __device_builtin_variable_blockIdx; 
#line 73
extern const dim3 __device_builtin_variable_blockDim; 
#line 74
extern const dim3 __device_builtin_variable_gridDim; 
#line 75
extern const int __device_builtin_variable_warpSize; 
#line 80
}
#line 185 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2/bin/../include\\cuda_runtime.h"
template< class T> static __inline ::cudaError_t 
#line 186
cudaLaunchKernel(const T *
#line 187
func, ::dim3 
#line 188
gridDim, ::dim3 
#line 189
blockDim, void **
#line 190
args, ::size_t 
#line 191
sharedMem = 0, ::cudaStream_t 
#line 192
stream = 0) 
#line 194
{ 
#line 195
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
#line 196
} 
#line 245
template< class T> static __inline ::cudaError_t 
#line 246
cudaLaunchCooperativeKernel(const T *
#line 247
func, ::dim3 
#line 248
gridDim, ::dim3 
#line 249
blockDim, void **
#line 250
args, ::size_t 
#line 251
sharedMem = 0, ::cudaStream_t 
#line 252
stream = 0) 
#line 254
{ 
#line 255
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
#line 256
} 
#line 283
template< class T> static __inline ::cudaError_t 
#line 284
cudaSetupArgument(T 
#line 285
arg, ::size_t 
#line 286
offset) 
#line 288
{ 
#line 289
return ::cudaSetupArgument((const void *)(&arg), sizeof(T), offset); 
#line 290
} 
#line 322
static __inline cudaError_t cudaEventCreate(cudaEvent_t *
#line 323
event, unsigned 
#line 324
flags) 
#line 326
{ 
#line 327
return ::cudaEventCreateWithFlags(event, flags); 
#line 328
} 
#line 385
static __inline cudaError_t cudaMallocHost(void **
#line 386
ptr, size_t 
#line 387
size, unsigned 
#line 388
flags) 
#line 390
{ 
#line 391
return ::cudaHostAlloc(ptr, size, flags); 
#line 392
} 
#line 394
template< class T> static __inline ::cudaError_t 
#line 395
cudaHostAlloc(T **
#line 396
ptr, ::size_t 
#line 397
size, unsigned 
#line 398
flags) 
#line 400
{ 
#line 401
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
#line 402
} 
#line 404
template< class T> static __inline ::cudaError_t 
#line 405
cudaHostGetDevicePointer(T **
#line 406
pDevice, void *
#line 407
pHost, unsigned 
#line 408
flags) 
#line 410
{ 
#line 411
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
#line 412
} 
#line 512
template< class T> static __inline ::cudaError_t 
#line 513
cudaMallocManaged(T **
#line 514
devPtr, ::size_t 
#line 515
size, unsigned 
#line 516
flags = 1) 
#line 518
{ 
#line 519
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
#line 520
} 
#line 600
template< class T> static __inline ::cudaError_t 
#line 601
cudaStreamAttachMemAsync(::cudaStream_t 
#line 602
stream, T *
#line 603
devPtr, ::size_t 
#line 604
length = 0, unsigned 
#line 605
flags = 4) 
#line 607
{ 
#line 608
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
#line 609
} 
#line 611
template< class T> __inline ::cudaError_t 
#line 612
cudaMalloc(T **
#line 613
devPtr, ::size_t 
#line 614
size) 
#line 616
{ 
#line 617
return ::cudaMalloc((void **)((void *)devPtr), size); 
#line 618
} 
#line 620
template< class T> static __inline ::cudaError_t 
#line 621
cudaMallocHost(T **
#line 622
ptr, ::size_t 
#line 623
size, unsigned 
#line 624
flags = 0) 
#line 626
{ 
#line 627
return cudaMallocHost((void **)((void *)ptr), size, flags); 
#line 628
} 
#line 630
template< class T> static __inline ::cudaError_t 
#line 631
cudaMallocPitch(T **
#line 632
devPtr, ::size_t *
#line 633
pitch, ::size_t 
#line 634
width, ::size_t 
#line 635
height) 
#line 637
{ 
#line 638
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
#line 639
} 
#line 676
template< class T> static __inline ::cudaError_t 
#line 677
cudaMemcpyToSymbol(const T &
#line 678
symbol, const void *
#line 679
src, ::size_t 
#line 680
count, ::size_t 
#line 681
offset = 0, ::cudaMemcpyKind 
#line 682
kind = cudaMemcpyHostToDevice) 
#line 684
{ 
#line 685
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
#line 686
} 
#line 728
template< class T> static __inline ::cudaError_t 
#line 729
cudaMemcpyToSymbolAsync(const T &
#line 730
symbol, const void *
#line 731
src, ::size_t 
#line 732
count, ::size_t 
#line 733
offset = 0, ::cudaMemcpyKind 
#line 734
kind = cudaMemcpyHostToDevice, ::cudaStream_t 
#line 735
stream = 0) 
#line 737
{ 
#line 738
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
#line 739
} 
#line 774
template< class T> static __inline ::cudaError_t 
#line 775
cudaMemcpyFromSymbol(void *
#line 776
dst, const T &
#line 777
symbol, ::size_t 
#line 778
count, ::size_t 
#line 779
offset = 0, ::cudaMemcpyKind 
#line 780
kind = cudaMemcpyDeviceToHost) 
#line 782
{ 
#line 783
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
#line 784
} 
#line 826
template< class T> static __inline ::cudaError_t 
#line 827
cudaMemcpyFromSymbolAsync(void *
#line 828
dst, const T &
#line 829
symbol, ::size_t 
#line 830
count, ::size_t 
#line 831
offset = 0, ::cudaMemcpyKind 
#line 832
kind = cudaMemcpyDeviceToHost, ::cudaStream_t 
#line 833
stream = 0) 
#line 835
{ 
#line 836
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
#line 837
} 
#line 860
template< class T> static __inline ::cudaError_t 
#line 861
cudaGetSymbolAddress(void **
#line 862
devPtr, const T &
#line 863
symbol) 
#line 865
{ 
#line 866
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
#line 867
} 
#line 890
template< class T> static __inline ::cudaError_t 
#line 891
cudaGetSymbolSize(::size_t *
#line 892
size, const T &
#line 893
symbol) 
#line 895
{ 
#line 896
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
#line 897
} 
#line 932
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 933
cudaBindTexture(::size_t *
#line 934
offset, const texture< T, dim, readMode>  &
#line 935
tex, const void *
#line 936
devPtr, const ::cudaChannelFormatDesc &
#line 937
desc, ::size_t 
#line 938
size = 4294967295U) 
#line 940
{ 
#line 941
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
#line 942
} 
#line 976
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 977
cudaBindTexture(::size_t *
#line 978
offset, const texture< T, dim, readMode>  &
#line 979
tex, const void *
#line 980
devPtr, ::size_t 
#line 981
size = 4294967295U) 
#line 983
{ 
#line 984
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
#line 985
} 
#line 1031
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1032
cudaBindTexture2D(::size_t *
#line 1033
offset, const texture< T, dim, readMode>  &
#line 1034
tex, const void *
#line 1035
devPtr, const ::cudaChannelFormatDesc &
#line 1036
desc, ::size_t 
#line 1037
width, ::size_t 
#line 1038
height, ::size_t 
#line 1039
pitch) 
#line 1041
{ 
#line 1042
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
#line 1043
} 
#line 1088
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1089
cudaBindTexture2D(::size_t *
#line 1090
offset, const texture< T, dim, readMode>  &
#line 1091
tex, const void *
#line 1092
devPtr, ::size_t 
#line 1093
width, ::size_t 
#line 1094
height, ::size_t 
#line 1095
pitch) 
#line 1097
{ 
#line 1098
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
#line 1099
} 
#line 1129
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1130
cudaBindTextureToArray(const texture< T, dim, readMode>  &
#line 1131
tex, ::cudaArray_const_t 
#line 1132
array, const ::cudaChannelFormatDesc &
#line 1133
desc) 
#line 1135
{ 
#line 1136
return ::cudaBindTextureToArray(&tex, array, &desc); 
#line 1137
} 
#line 1166
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1167
cudaBindTextureToArray(const texture< T, dim, readMode>  &
#line 1168
tex, ::cudaArray_const_t 
#line 1169
array) 
#line 1171
{ 
#line 1172
::cudaChannelFormatDesc desc; 
#line 1173
::cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
#line 1175
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
#line 1176
} 
#line 1206
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1207
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
#line 1208
tex, ::cudaMipmappedArray_const_t 
#line 1209
mipmappedArray, const ::cudaChannelFormatDesc &
#line 1210
desc) 
#line 1212
{ 
#line 1213
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
#line 1214
} 
#line 1243
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1244
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
#line 1245
tex, ::cudaMipmappedArray_const_t 
#line 1246
mipmappedArray) 
#line 1248
{ 
#line 1249
::cudaChannelFormatDesc desc; 
#line 1250
::cudaArray_t levelArray; 
#line 1251
::cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
#line 1253
if (err != (cudaSuccess)) { 
#line 1254
return err; 
#line 1255
}  
#line 1256
err = ::cudaGetChannelDesc(&desc, levelArray); 
#line 1258
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
#line 1259
} 
#line 1284
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1285
cudaUnbindTexture(const texture< T, dim, readMode>  &
#line 1286
tex) 
#line 1288
{ 
#line 1289
return ::cudaUnbindTexture(&tex); 
#line 1290
} 
#line 1318
template< class T, int dim, cudaTextureReadMode readMode> static __inline ::cudaError_t 
#line 1319
cudaGetTextureAlignmentOffset(::size_t *
#line 1320
offset, const texture< T, dim, readMode>  &
#line 1321
tex) 
#line 1323
{ 
#line 1324
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
#line 1325
} 
#line 1370
template< class T> static __inline ::cudaError_t 
#line 1371
cudaFuncSetCacheConfig(T *
#line 1372
func, ::cudaFuncCache 
#line 1373
cacheConfig) 
#line 1375
{ 
#line 1376
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
#line 1377
} 
#line 1379
template< class T> static __inline ::cudaError_t 
#line 1380
cudaFuncSetSharedMemConfig(T *
#line 1381
func, ::cudaSharedMemConfig 
#line 1382
config) 
#line 1384
{ 
#line 1385
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
#line 1386
} 
#line 1415
template< class T> __inline ::cudaError_t 
#line 1416
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
#line 1417
numBlocks, T 
#line 1418
func, int 
#line 1419
blockSize, ::size_t 
#line 1420
dynamicSMemSize) 
#line 1421
{ 
#line 1422
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
#line 1423
} 
#line 1466
template< class T> __inline ::cudaError_t 
#line 1467
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
#line 1468
numBlocks, T 
#line 1469
func, int 
#line 1470
blockSize, ::size_t 
#line 1471
dynamicSMemSize, unsigned 
#line 1472
flags) 
#line 1473
{ 
#line 1474
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
#line 1475
} 
#line 1480
class __cudaOccupancyB2DHelper { 
#line 1481
size_t n; 
#line 1483
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
#line 1484
size_t operator()(int) 
#line 1485
{ 
#line 1486
return n; 
#line 1487
} 
#line 1488
}; 
#line 1535
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 1536
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
#line 1537
minGridSize, int *
#line 1538
blockSize, T 
#line 1539
func, UnaryFunction 
#line 1540
blockSizeToDynamicSMemSize, int 
#line 1541
blockSizeLimit = 0, unsigned 
#line 1542
flags = 0) 
#line 1543
{ 
#line 1544
::cudaError_t status; 
#line 1547
int device; 
#line 1548
::cudaFuncAttributes attr; 
#line 1551
int maxThreadsPerMultiProcessor; 
#line 1552
int warpSize; 
#line 1553
int devMaxThreadsPerBlock; 
#line 1554
int multiProcessorCount; 
#line 1555
int funcMaxThreadsPerBlock; 
#line 1556
int occupancyLimit; 
#line 1557
int granularity; 
#line 1560
int maxBlockSize = 0; 
#line 1561
int numBlocks = 0; 
#line 1562
int maxOccupancy = 0; 
#line 1565
int blockSizeToTryAligned; 
#line 1566
int blockSizeToTry; 
#line 1567
int blockSizeLimitAligned; 
#line 1568
int occupancyInBlocks; 
#line 1569
int occupancyInThreads; 
#line 1570
::size_t dynamicSMemSize; 
#line 1576
if (((!minGridSize) || (!blockSize)) || (!func)) { 
#line 1577
return cudaErrorInvalidValue; 
#line 1578
}  
#line 1584
status = ::cudaGetDevice(&device); 
#line 1585
if (status != (cudaSuccess)) { 
#line 1586
return status; 
#line 1587
}  
#line 1589
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
#line 1593
if (status != (cudaSuccess)) { 
#line 1594
return status; 
#line 1595
}  
#line 1597
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
#line 1601
if (status != (cudaSuccess)) { 
#line 1602
return status; 
#line 1603
}  
#line 1605
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
#line 1609
if (status != (cudaSuccess)) { 
#line 1610
return status; 
#line 1611
}  
#line 1613
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
#line 1617
if (status != (cudaSuccess)) { 
#line 1618
return status; 
#line 1619
}  
#line 1621
status = cudaFuncGetAttributes(&attr, func); 
#line 1622
if (status != (cudaSuccess)) { 
#line 1623
return status; 
#line 1624
}  
#line 1626
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
#line 1632
occupancyLimit = maxThreadsPerMultiProcessor; 
#line 1633
granularity = warpSize; 
#line 1635
if (blockSizeLimit == 0) { 
#line 1636
blockSizeLimit = devMaxThreadsPerBlock; 
#line 1637
}  
#line 1639
if (devMaxThreadsPerBlock < blockSizeLimit) { 
#line 1640
blockSizeLimit = devMaxThreadsPerBlock; 
#line 1641
}  
#line 1643
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
#line 1644
blockSizeLimit = funcMaxThreadsPerBlock; 
#line 1645
}  
#line 1647
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
#line 1649
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
#line 1653
if (blockSizeLimit < blockSizeToTryAligned) { 
#line 1654
blockSizeToTry = blockSizeLimit; 
#line 1655
} else { 
#line 1656
blockSizeToTry = blockSizeToTryAligned; 
#line 1657
}  
#line 1659
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
#line 1661
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
#line 1668
if (status != (cudaSuccess)) { 
#line 1669
return status; 
#line 1670
}  
#line 1672
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
#line 1674
if (occupancyInThreads > maxOccupancy) { 
#line 1675
maxBlockSize = blockSizeToTry; 
#line 1676
numBlocks = occupancyInBlocks; 
#line 1677
maxOccupancy = occupancyInThreads; 
#line 1678
}  
#line 1682
if (occupancyLimit == maxOccupancy) { 
#line 1683
break; 
#line 1684
}  
#line 1685
}  
#line 1693
(*minGridSize) = (numBlocks * multiProcessorCount); 
#line 1694
(*blockSize) = maxBlockSize; 
#line 1696
return status; 
#line 1697
} 
#line 1730
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 1731
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
#line 1732
minGridSize, int *
#line 1733
blockSize, T 
#line 1734
func, UnaryFunction 
#line 1735
blockSizeToDynamicSMemSize, int 
#line 1736
blockSizeLimit = 0) 
#line 1737
{ 
#line 1738
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
#line 1739
} 
#line 1775
template< class T> static __inline ::cudaError_t 
#line 1776
cudaOccupancyMaxPotentialBlockSize(int *
#line 1777
minGridSize, int *
#line 1778
blockSize, T 
#line 1779
func, ::size_t 
#line 1780
dynamicSMemSize = 0, int 
#line 1781
blockSizeLimit = 0) 
#line 1782
{ 
#line 1783
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
#line 1784
} 
#line 1834
template< class T> static __inline ::cudaError_t 
#line 1835
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
#line 1836
minGridSize, int *
#line 1837
blockSize, T 
#line 1838
func, ::size_t 
#line 1839
dynamicSMemSize = 0, int 
#line 1840
blockSizeLimit = 0, unsigned 
#line 1841
flags = 0) 
#line 1842
{ 
#line 1843
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
#line 1844
} 
#line 1885
template< class T> static __inline ::cudaError_t 
#line 1886
cudaLaunch(T *
#line 1887
func) 
#line 1889
{ 
#line 1890
return ::cudaLaunch((const void *)func); 
#line 1891
} 
#line 1922
template< class T> __inline ::cudaError_t 
#line 1923
cudaFuncGetAttributes(::cudaFuncAttributes *
#line 1924
attr, T *
#line 1925
entry) 
#line 1927
{ 
#line 1928
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
#line 1929
} 
#line 1967
template< class T> static __inline ::cudaError_t 
#line 1968
cudaFuncSetAttribute(T *
#line 1969
entry, ::cudaFuncAttribute 
#line 1970
attr, int 
#line 1971
value) 
#line 1973
{ 
#line 1974
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
#line 1975
} 
#line 1997
template< class T, int dim> static __inline ::cudaError_t 
#line 1998
cudaBindSurfaceToArray(const surface< T, dim>  &
#line 1999
surf, ::cudaArray_const_t 
#line 2000
array, const ::cudaChannelFormatDesc &
#line 2001
desc) 
#line 2003
{ 
#line 2004
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
#line 2005
} 
#line 2026
template< class T, int dim> static __inline ::cudaError_t 
#line 2027
cudaBindSurfaceToArray(const surface< T, dim>  &
#line 2028
surf, ::cudaArray_const_t 
#line 2029
array) 
#line 2031
{ 
#line 2032
::cudaChannelFormatDesc desc; 
#line 2033
::cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
#line 2035
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
#line 2036
} 
#line 2050 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2/bin/../include\\cuda_runtime.h"
#pragma warning(pop)
#line 43 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
#line 45
const char *info_simulate = ("INFO:simulate[MSVC]"); 
#line 235 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((9 / 10000000) % 10)), (('0') + ((9 / 1000000) % 10)), (('0') + ((9 / 100000) % 10)), (('0') + ((9 / 10000) % 10)), (('0') + ((9 / 1000) % 10)), (('0') + ((9 / 100) % 10)), (('0') + ((9 / 10) % 10)), (('0') + (9 % 10)), '.', (('0') + ((2 / 10000000) % 10)), (('0') + ((2 / 1000000) % 10)), (('0') + ((2 / 100000) % 10)), (('0') + ((2 / 10000) % 10)), (('0') + ((2 / 1000) % 10)), (('0') + ((2 / 100) % 10)), (('0') + ((2 / 10) % 10)), (('0') + (2 % 10)), '.', (('0') + ((148 / 10000000) % 10)), (('0') + ((148 / 1000000) % 10)), (('0') + ((148 / 100000) % 10)), (('0') + ((148 / 10000) % 10)), (('0') + ((148 / 1000) % 10)), (('0') + ((148 / 100) % 10)), (('0') + ((148 / 10) % 10)), (('0') + (148 % 10)), ']', '\000'}; 
#line 262 "CMakeCUDACompilerId.cu"
const char info_simulate_version[] = {'I', 'N', 'F', 'O', ':', 's', 'i', 'm', 'u', 'l', 'a', 't', 'e', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + (((1916 / 100) / 10000000) % 10)), (('0') + (((1916 / 100) / 1000000) % 10)), (('0') + (((1916 / 100) / 100000) % 10)), (('0') + (((1916 / 100) / 10000) % 10)), (('0') + (((1916 / 100) / 1000) % 10)), (('0') + (((1916 / 100) / 100) % 10)), (('0') + (((1916 / 100) / 10) % 10)), (('0') + ((1916 / 100) % 10)), '.', (('0') + (((1916 % 100) / 10000000) % 10)), (('0') + (((1916 % 100) / 1000000) % 10)), (('0') + (((1916 % 100) / 100000) % 10)), (('0') + (((1916 % 100) / 10000) % 10)), (('0') + (((1916 % 100) / 1000) % 10)), (('0') + (((1916 % 100) / 100) % 10)), (('0') + (((1916 % 100) / 10) % 10)), (('0') + ((1916 % 100) % 10)), ']', '\000'}; 
#line 282 "CMakeCUDACompilerId.cu"
const char *info_platform = ("INFO:platform[Windows]"); 
#line 283
const char *info_arch = ("INFO:arch[x64]"); 
#line 288
const char *info_language_dialect_default = ("INFO:dialect_default[98]"); 
#line 304 "CMakeCUDACompilerId.cu"
int main(int argc, char *argv[]) 
#line 305
{ 
#line 306
int require = 0; 
#line 307
require += (info_compiler[argc]); 
#line 308
require += (info_platform[argc]); 
#line 310
require += (info_version[argc]); 
#line 313 "CMakeCUDACompilerId.cu"
require += (info_simulate[argc]); 
#line 316 "CMakeCUDACompilerId.cu"
require += (info_simulate_version[argc]); 
#line 318 "CMakeCUDACompilerId.cu"
require += (info_language_dialect_default[argc]); 
#line 319
(void)argv; 
#line 320
return require; 
#line 321
} 
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__27_CMakeCUDACompilerId_cpp1_ii_bd57c623
#pragma pack()
#define __NV_USE_NEW_KERNEL_LAUNCH 1
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
