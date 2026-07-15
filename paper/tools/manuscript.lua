local function normalize_image(image)
  if image.src:match("^figures/") then
    image.src = "../../" .. image.src
  end
  return image
end

-- Keep image paths valid when LaTeX is compiled from paper/latex/neurips_2026.
function Image(image)
  return normalize_image(image)
end

-- The four library renders are adjacent Markdown images; retain them as a stable 2x2 panel.
function Para(paragraph)
  local images = {}
  for _, inline in ipairs(paragraph.content) do
    if inline.t == "Image" then
      table.insert(images, normalize_image(inline))
    elseif inline.t ~= "Space" and inline.t ~= "SoftBreak" then
      return nil
    end
  end
  if #images ~= 4 then
    return nil
  end
  for _, image in ipairs(images) do
    image.attributes.width = "47%"
    image.attributes.height = "35%"
  end
  return pandoc.Para({
    images[1], pandoc.Space(), images[2], pandoc.LineBreak(),
    images[3], pandoc.Space(), images[4]
  })
end

-- GFM has no column-width syntax. Equal explicit widths make every table wrap
-- inside the venue's text block instead of emitting natural-width longtables.
function Table(tbl)
  local width = 1.0 / #tbl.colspecs
  for index, colspec in ipairs(tbl.colspecs) do
    tbl.colspecs[index] = { colspec[1], width }
  end
  return tbl
end

-- xurl's \path permits line breaks in long config names without shrinking type.
function Code(code)
  if #code.text >= 18 and code.text:match("^[A-Za-z0-9_./-]+$") then
    return pandoc.RawInline("latex", "\\path{" .. code.text .. "}")
  end
  return nil
end
