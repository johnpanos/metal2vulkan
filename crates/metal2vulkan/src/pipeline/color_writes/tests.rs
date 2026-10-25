//! [`color_writes`] of the 32-bit float attachments: a blend of factors other than one and zero
//! is the device's arithmetic (`ChannelWrite::Device`, which records
//! `PrecisionNote::DeviceColorConversion`) where the device does not store Metal's color writes,
//! and nothing otherwise.

use super::*;
use crate::metal::{BlendFactor, PixelFormat};

/// Attachment 0 of `pf`, blended source-over where `blend` is `Some(true)`, with factors one and
/// zero where `Some(false)`, not blended where `None`.
fn attachment(pf: PixelFormat, blend: Option<bool>) -> ColorAttachmentDescriptor {
    let mut c = ColorAttachmentDescriptor::new(pf);
    if let Some(over) = blend {
        c.blending_enabled = true;
        if over {
            c.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            c.source_alpha_blend_factor = BlendFactor::SourceAlpha;
            c.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        }
    }
    c
}

/// Every committed profile's map for the 32-bit float formats: each channel the device's under a
/// source-over blend, unless the profile stores Metal's color writes (MoltenVK), and no entry
/// without a blend or under factors one and zero; an integer format takes none either way.
#[test]
fn a_blended_32_bit_float_channel_is_the_devices() {
    let mut device_blends = 0;
    for (stem, dev) in crate::metal::format::tests::committed_devices() {
        let p = dev.profile();
        let metal = p
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalColorWrites);
        for (pf, channels) in [
            (PixelFormat::R32Float, 1),
            (PixelFormat::RG32Float, 2),
            (PixelFormat::RGBA32Float, 4),
        ] {
            let map = |blend| {
                let colors = [Some(attachment(pf, blend))];
                color_writes(
                    [0],
                    &colors,
                    (false, TextureWriteRoundingMode::TowardZero),
                    p,
                )
            };
            let blended = map(Some(true));
            if metal {
                assert!(blended.is_empty(), "{stem} {pf:?}: {blended:?}");
            } else {
                let mut want = [None; 4];
                for channel in want.iter_mut().take(channels) {
                    *channel = Some(ChannelWrite::Device);
                }
                assert_eq!(blended.get(&0), Some(&want), "{stem} {pf:?}");
                device_blends += 1;
            }
            assert!(map(None).is_empty(), "{stem} {pf:?} unblended");
            assert!(map(Some(false)).is_empty(), "{stem} {pf:?} one and zero");
        }
        let colors = [Some(attachment(PixelFormat::RGBA32Uint, None))];
        let integer = color_writes(
            [0],
            &colors,
            (false, TextureWriteRoundingMode::TowardZero),
            p,
        );
        assert!(integer.is_empty(), "{stem} RGBA32Uint");
    }
    assert!(
        device_blends > 0,
        "no committed profile blends on the device"
    );
}

/// `ColorWritePolicy`: on every committed profile that does not store Metal's color writes, an
/// unblended 8-bit unorm, 8-bit sRGB or 16-bit float attachment's channels are the device's
/// conversion under the default policy (no rounding emitted, `PrecisionNote::DeviceColorConversion`
/// recorded), and rounded in the shader under `ColorWritePolicy::Exact` (a rule each channel or
/// the draw's selected one); a profile that stores Metal's codes maps the same under either.
#[test]
fn a_color_write_is_the_devices_conversion_unless_the_policy_asks_for_metal_s() {
    use crate::device::{ColorWritePolicy, Device, DevicePolicy};
    let mut checked = 0;
    for (stem, desc) in crate::metal::format::tests::committed_descriptions() {
        let device = |color_writes| {
            let policy = DevicePolicy {
                color_writes,
                ..DevicePolicy::default()
            };
            Device::new(desc.clone(), policy).expect("device derives")
        };
        let (fast, exact) = (
            device(ColorWritePolicy::Device),
            device(ColorWritePolicy::Exact),
        );
        let metal = fast
            .profile()
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalColorWrites);
        for pf in [
            PixelFormat::BGRA8Unorm,
            PixelFormat::RGBA8Unorm,
            PixelFormat::BGRA8UnormSRGB,
            PixelFormat::RGBA16Float,
        ] {
            let map = |dev: &Device| {
                let colors = [Some(attachment(pf, None))];
                color_writes(
                    [0],
                    &colors,
                    (false, TextureWriteRoundingMode::TowardZero),
                    dev.profile(),
                )
            };
            let (f, e) = (map(&fast), map(&exact));
            if metal {
                assert_eq!(f, e, "{stem} {pf:?}");
                continue;
            }
            assert_eq!(
                f.get(&0),
                Some(&[Some(ChannelWrite::Device); 4]),
                "{stem} {pf:?} default"
            );
            let rounds = e.get(&0).is_some_and(|rules| {
                rules
                    .iter()
                    .flatten()
                    .all(|r| !matches!(r, ChannelWrite::Device))
            });
            assert!(rounds, "{stem} {pf:?} exact: {e:?}");
            checked += 1;
        }
    }
    assert!(checked > 0, "no committed profile converts on the device");
}
