#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Text labels anchored to world positions and drawn at a fixed pixel size
 * over the scene.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/buffer/SimpleCollector.hpp>
#include <solvcon/pilot/visual/RMaterial.hpp>

#include <rhi/qrhi.h>

#include <QColor>
#include <QImage>
#include <QMatrix4x4>
#include <QPointF>
#include <QRect>
#include <QString>
#include <QVector3D>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace solvcon
{

/**
 * @brief A set of named text labels, each anchored to a world position.
 *
 * Every label text is painted once into a shared texture atlas, repainted only
 * when a text, the device pixel ratio, or the palette text color changes. Each
 * frame projects the anchors to the screen and draws one textured quad per
 * label at a fixed pixel size, so a label stays readable at any zoom. An
 * anchor behind the eye or outside the viewport draws nothing. The quads
 * draw without the depth test, after the scene.
 *
 * @ingroup group_domain
 */
class RLabelOverlay
{

public:

    RLabelOverlay();
    ~RLabelOverlay();

    void setVisible(bool visible) { m_visible = visible; }
    bool isVisible() const { return m_visible; }

    /**
     * Add the label @p name, or replace it when the name exists. The text
     * starts at the anchor @p position (world units) shifted by @p offset in
     * logical pixels (x right, y down), vertically centered on that point.
     */
    void add(std::string const & name, std::string const & text, QVector3D const & position, QPointF const & offset);
    void remove(std::string const & name);
    void clear();
    std::vector<std::string> names() const;

    /**
     * Prepare the frame. @p view_proj is the view-projection in the OpenGL
     * clip convention (no QRhi clip-space correction), @p pixel_size the
     * render-target size in device pixels, and @p dpr the device pixel ratio.
     */
    void update(
        QRhi * rhi,
        QRhiRenderPassDescriptor * rpdesc,
        int sample_count,
        QSize pixel_size,
        float dpr,
        QMatrix4x4 const & view_proj,
        QRhiResourceUpdateBatch * batch);
    void draw(QRhiCommandBuffer * cb);
    void release();

private:

    /// A label and its place in the atlas, in device pixels.
    struct Entry
    {
        QString text;
        QVector3D position;
        QPointF offset; ///< Logical pixels, x right and y down.
        QRect atlas_rect;
    }; /* end struct Entry */

    QImage renderAtlas(float dpr, QColor const & color);
    void prepare(QRhi * rhi, QRhiRenderPassDescriptor * rpdesc, int sample_count, QRhiResourceUpdateBatch * batch);
    void uploadAtlas(QRhi * rhi, QImage const & atlas, QRhiResourceUpdateBatch * batch);

    // TODO: Replace the STL map once the label API settles.
    std::map<std::string, Entry> m_entries;
    SimpleCollector<float> m_vertices; ///< Six (x, y, z, u, v) vertices per drawn label.

    bool m_visible = true;
    bool m_ready = false;
    bool m_dirty = true;
    float m_dpr = 0.0f; ///< The device pixel ratio the atlas was painted at.
    QColor m_color; ///< The text color the atlas was painted in.
    uint32_t m_draw_count = 0; ///< Vertices to draw this frame.

    std::unique_ptr<RMaterial> m_material;
    std::unique_ptr<QRhiBuffer> m_vbuf;
    std::unique_ptr<QRhiBuffer> m_ubuf;
    std::unique_ptr<QRhiTexture> m_texture;
    std::unique_ptr<QRhiSampler> m_sampler;
    std::unique_ptr<QRhiShaderResourceBindings> m_srb;
    std::unique_ptr<QRhiGraphicsPipeline> m_pipeline;

}; /* end class RLabelOverlay */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
