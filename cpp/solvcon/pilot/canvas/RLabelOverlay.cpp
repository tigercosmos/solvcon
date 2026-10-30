/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/canvas/RLabelOverlay.hpp> // Must be the first include.

#include <QFont>
#include <QFontMetricsF>
#include <QGuiApplication>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QPen>

#include <algorithm>
#include <cmath>

namespace solvcon
{

namespace
{

constexpr float FONT_PIXEL_SIZE = 13.0f; ///< Logical pixels.
constexpr float HALO_WIDTH = 3.0f; ///< Logical pixels.
constexpr int ATLAS_MIN_WIDTH = 1024; ///< Device pixels.
constexpr int FLOATS_PER_VERTEX = 5;

QRhiShaderResourceBinding uniformBinding(QRhiBuffer * ubuf)
{
    return QRhiShaderResourceBinding::uniformBuffer(
        0, QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage, ubuf);
}

QRhiShaderResourceBinding textureBinding(QRhiTexture * texture, QRhiSampler * sampler)
{
    return QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, texture, sampler);
}

} /* end namespace */

RLabelOverlay::RLabelOverlay() = default;

RLabelOverlay::~RLabelOverlay() = default;

void RLabelOverlay::add(
    std::string const & name, std::string const & text, QVector3D const & position, QPointF const & offset)
{
    Entry & entry = m_entries[name];
    entry.position = position;
    entry.offset = offset;
    QString const qtext = QString::fromStdString(text);
    if (entry.text != qtext || entry.atlas_rect.isNull())
    {
        entry.text = qtext;
        m_dirty = true;
    }
}

void RLabelOverlay::remove(std::string const & name)
{
    m_entries.erase(name);
}

void RLabelOverlay::clear()
{
    m_entries.clear();
}

std::vector<std::string> RLabelOverlay::names() const
{
    std::vector<std::string> ret;
    ret.reserve(m_entries.size());
    for (auto const & entry : m_entries)
    {
        ret.push_back(entry.first);
    }
    return ret;
}

QImage RLabelOverlay::renderAtlas(float dpr, QColor const & color)
{
    QFont font;
    font.setPixelSize(std::max(1, static_cast<int>(std::lround(FONT_PIXEL_SIZE * dpr))));
    font.setBold(true);
    QFontMetricsF const metrics(font);
    int const pad = static_cast<int>(std::ceil(HALO_WIDTH * dpr));
    int const cell_height = static_cast<int>(std::ceil(metrics.height())) + 2 * pad;
    auto cell_width = [&metrics, pad](QString const & text)
    { return static_cast<int>(std::ceil(metrics.horizontalAdvance(text))) + 2 * pad; };

    // Shelf-pack the label cells left to right, starting a new row when one
    // would cross the atlas width.
    int atlas_width = ATLAS_MIN_WIDTH;
    for (auto const & entry : m_entries)
    {
        atlas_width = std::max(atlas_width, cell_width(entry.second.text));
    }
    int x = 0;
    int y = 0;
    for (auto & entry : m_entries)
    {
        int const width = cell_width(entry.second.text);
        if (x + width > atlas_width)
        {
            x = 0;
            y += cell_height;
        }
        entry.second.atlas_rect = QRect(x, y, width, cell_height);
        x += width;
    }

    QImage image(atlas_width, y + cell_height, QImage::Format_RGBA8888);
    image.fill(Qt::transparent);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);

    // A halo in the window color keeps the text legible over the scene lines.
    QPen halo(QGuiApplication::palette().color(QPalette::Window), HALO_WIDTH * dpr);
    halo.setJoinStyle(Qt::RoundJoin);
    for (auto const & entry : m_entries)
    {
        QRect const & rect = entry.second.atlas_rect;
        QPainterPath path;
        path.addText(QPointF(rect.x() + pad, rect.y() + pad + metrics.ascent()), font, entry.second.text);
        painter.strokePath(path, halo);
        painter.fillPath(path, color);
    }
    painter.end();
    return image;
}

void RLabelOverlay::prepare(
    QRhi * rhi, QRhiRenderPassDescriptor * rpdesc, int sample_count, QRhiResourceUpdateBatch * batch)
{
    m_ubuf.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 64 + 16));
    m_ubuf->create();
    float const white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    batch->updateDynamicBuffer(m_ubuf.get(), 64, 16, white);

    m_sampler.reset(rhi->newSampler(
        QRhiSampler::Linear,
        QRhiSampler::Linear,
        QRhiSampler::None,
        QRhiSampler::ClampToEdge,
        QRhiSampler::ClampToEdge));
    m_sampler->create();

    m_texture.reset(rhi->newTexture(QRhiTexture::RGBA8, QSize(1, 1)));
    m_texture->create();

    m_srb.reset(rhi->newShaderResourceBindings());
    m_srb->setBindings({uniformBinding(m_ubuf.get()), textureBinding(m_texture.get(), m_sampler.get())});
    m_srb->create();

    QRhiVertexInputLayout layout;
    layout.setBindings({{FLOATS_PER_VERTEX * sizeof(float)}});
    layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float2, 3 * sizeof(float)},
    });

    m_material = std::make_unique<RMaterial>(RMaterial::Kind::Textured);
    m_pipeline.reset(m_material->buildPipeline(
        rhi, m_srb.get(), rpdesc, layout, QRhiGraphicsPipeline::Triangles, sample_count, false, true));

    m_ready = (nullptr != m_pipeline);
    m_dirty = true;
}

void RLabelOverlay::uploadAtlas(QRhi * rhi, QImage const & atlas, QRhiResourceUpdateBatch * batch)
{
    if (m_texture->pixelSize() != atlas.size())
    {
        m_texture.reset(rhi->newTexture(QRhiTexture::RGBA8, atlas.size()));
        m_texture->create();
        m_srb->setBindings({uniformBinding(m_ubuf.get()), textureBinding(m_texture.get(), m_sampler.get())});
        m_srb->create();
    }
    batch->uploadTexture(m_texture.get(), atlas);
}

void RLabelOverlay::update(
    QRhi * rhi,
    QRhiRenderPassDescriptor * rpdesc,
    int sample_count,
    QSize pixel_size,
    float dpr,
    QMatrix4x4 const & view_proj,
    QRhiResourceUpdateBatch * batch)
{
    m_draw_count = 0;
    if (!m_visible || m_entries.empty() || pixel_size.isEmpty())
    {
        return;
    }
    if (!m_ready)
    {
        prepare(rhi, rpdesc, sample_count, batch);
        if (!m_ready)
        {
            return;
        }
    }

    QColor const color = QGuiApplication::palette().color(QPalette::WindowText);
    if (m_dirty || dpr != m_dpr || color != m_color)
    {
        uploadAtlas(rhi, renderAtlas(dpr, color), batch);
        m_dpr = dpr;
        m_color = color;
        m_dirty = false;
    }

    float const width = static_cast<float>(pixel_size.width());
    float const height = static_cast<float>(pixel_size.height());
    float const atlas_width = static_cast<float>(m_texture->pixelSize().width());
    float const atlas_height = static_cast<float>(m_texture->pixelSize().height());
    m_vertices.clear();
    for (auto const & entry : m_entries)
    {
        Entry const & label = entry.second;
        QVector4D const clip = view_proj * QVector4D(label.position, 1.0f);
        if (clip.w() <= 0.0f)
        {
            continue;
        }
        QVector3D const ndc = clip.toVector3D() / clip.w();
        if (std::abs(ndc.x()) > 1.0f || std::abs(ndc.y()) > 1.0f || std::abs(ndc.z()) > 1.0f)
        {
            continue;
        }

        // Snap the quad to whole device pixels (origin at the top left) so the
        // atlas texels map one to one onto the screen.
        QRect const & rect = label.atlas_rect;
        float const rect_width = static_cast<float>(rect.width());
        float const rect_height = static_cast<float>(rect.height());
        float const left = std::round(0.5f * (ndc.x() + 1.0f) * width + label.offset.x() * dpr);
        float const top = std::round(0.5f * (1.0f - ndc.y()) * height + label.offset.y() * dpr - 0.5f * rect_height);
        float const x0 = 2.0f * left / width - 1.0f;
        float const x1 = 2.0f * (left + rect_width) / width - 1.0f;
        float const y0 = 1.0f - 2.0f * top / height;
        float const y1 = 1.0f - 2.0f * (top + rect_height) / height;
        float const u0 = static_cast<float>(rect.x()) / atlas_width;
        float const u1 = static_cast<float>(rect.x() + rect.width()) / atlas_width;
        float const v0 = static_cast<float>(rect.y()) / atlas_height;
        float const v1 = static_cast<float>(rect.y() + rect.height()) / atlas_height;
        // clang-format off
        float const quad[6 * FLOATS_PER_VERTEX] = {
            x0, y1, 0.0f, u0, v1,
            x1, y1, 0.0f, u1, v1,
            x1, y0, 0.0f, u1, v0,
            x0, y1, 0.0f, u0, v1,
            x1, y0, 0.0f, u1, v0,
            x0, y0, 0.0f, u0, v0,
        };
        // clang-format on
        for (float const value : quad)
        {
            m_vertices.push_back(value);
        }
    }
    if (m_vertices.empty())
    {
        return;
    }

    quint32 const bytes = static_cast<quint32>(m_vertices.size() * sizeof(float));
    if (!m_vbuf || m_vbuf->size() < bytes)
    {
        m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::VertexBuffer, bytes));
        m_vbuf->create();
    }
    batch->updateDynamicBuffer(m_vbuf.get(), 0, bytes, m_vertices.data());

    QMatrix4x4 const mvp = rhi->clipSpaceCorrMatrix();
    batch->updateDynamicBuffer(m_ubuf.get(), 0, 64, mvp.constData());

    m_draw_count = static_cast<uint32_t>(m_vertices.size() / FLOATS_PER_VERTEX);
}

void RLabelOverlay::draw(QRhiCommandBuffer * cb)
{
    if (!m_visible || 0 == m_draw_count)
    {
        return;
    }
    cb->setGraphicsPipeline(m_pipeline.get());
    cb->setShaderResources(m_srb.get());
    QRhiCommandBuffer::VertexInput const quad_in(m_vbuf.get(), 0);
    cb->setVertexInput(0, 1, &quad_in);
    cb->draw(m_draw_count);
}

void RLabelOverlay::release()
{
    m_pipeline.reset();
    m_srb.reset();
    m_texture.reset();
    m_sampler.reset();
    m_ubuf.reset();
    m_vbuf.reset();
    m_material.reset();
    m_ready = false;
    m_dirty = true;
    m_draw_count = 0;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
