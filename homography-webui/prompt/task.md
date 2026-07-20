# Refine the Defect Grouping Logic: Duplicate Detection vs Multi-Frame Stitching

The current **Group Defects** function is incorrectly handling defects detected across multiple frames. The grouping logic needs to distinguish between:

1. **Duplicate detections of the same defect** (the same physical area detected repeatedly in multiple frames).
2. **A single large defect spanning multiple frames** (different portions of the same physical defect captured across adjacent frames).

These two cases must be handled differently.

---

# Core Differentiation Logic

The **primary factor for deciding between duplicates and multi-frame defects is mask shape similarity**.

## Case 1: Duplicate Detection (Same Defect Appearing in Multiple Frames)

If two detections have:

- Very similar mask shape.
- Similar orientation.
- Similar dimensions.
- Occur in sequential or nearby frames.
- Are spatially close after projection (approximately within 1–2 metres).

Then they should be considered **duplicate detections of the same physical defect**.

### Duplicate Handling

When duplicates are identified:

- Collapse them into a single defect object.
- Keep only one mask.
- **Do not merge the masks together.**
- Use the mask from the frame where the defect appears **lowest in the image** (closest to the camera).
  - This frame should provide the highest-resolution and most complete segmentation.
- Remove the duplicate instances from:
  - the defect list,
  - the map,
  - all downstream processing.

---

# Case 2: Large Defect Spanning Multiple Frames

If two detections:

- Are in sequential frames.
- Are spatially connected or overlapping in the real world.
- Appear to continue from one frame into the next.
- But have **different mask shapes**.

Then they should be treated as **different sections of the same large defect**.

The difference in mask shape is the key indicator:

- Same shape + close together → likely duplicate detection.
- Different shape + connected/continuous position → likely one large defect spanning frames.

### Multi-Frame Stitching Behaviour

When combining these defects:

- Preserve the original masks exactly.
- **Do not rerun segmentation.**
- **Do not recalculate masks.**
- **Do not simplify polygons.**
- Simply stitch the existing masks together at the frame boundary.
- Maintain all original segmentation detail.

The output should be:

- One defect ID.
- One combined mask.
- One combined measurement.
- One entry in the defect list.
- One mapped defect location.

---

# Implementation Requirements

Before modifying the code:

1. Review the current grouping implementation.
2. Explain:
   - how duplicate detection currently works,
   - how multi-frame stitching currently works,
   - why the current logic incorrectly merges or separates defects.
3. Propose the updated decision process before implementing.

The grouping pipeline should follow this general order:

1. Compare nearby defects from sequential frames.
2. Calculate mask similarity/shape similarity.
3. If masks are highly similar:
   - classify as duplicate,
   - retain only the best mask (lowest in frame).
4. If masks are different but spatially continuous:
   - classify as a multi-frame defect,
   - stitch masks together.
5. Ensure all grouped defects remain consistent across:
   - defect list,
   - map display,
   - area calculations,
   - depth calculations,
   - exports.

The final grouping behaviour should preserve segmentation accuracy and avoid unnecessary mask modification.