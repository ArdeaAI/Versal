local stringify = pandoc.utils.stringify

local image_sizes = {
  ["figures/system_lifecycle.pdf"] = {width = "0.98\\linewidth"},
  ["figures/overmind_structural.png"] = {height = "2.65in"},
  ["figures/clean_canary.pdf"] = {width = "0.97\\linewidth"},
  ["figures/persistence_xor.pdf"] = {width = "0.95\\linewidth"},
}

local title_block = pandoc.RawBlock("latex", [[
\begin{center}
\rule{\linewidth}{1.15pt}
\vspace{0.28in}

{\fontsize{19}{22}\selectfont\bfseries Versal: Versatile Evolution of Reusable Structure for Adaptive Learning\par}

\vspace{0.22in}
{\normalsize J. R. M. Gardner\par}
{\small Ardea AI, with Apart Research\par}
{\small \href{mailto:john@ardea.io}{john@ardea.io}\par}
{\small \href{https://orcid.org/0009-0000-9879-9882}{ORCID: 0009-0000-9879-9882}\par}

\vspace{0.14in}
{\footnotesize\itshape Research conducted at the Digital Minds Research Sprint, August 2026.\par}
\vspace{0.18in}
{\large\bfseries Abstract\par}
\vspace{0.04in}
\end{center}
]])

function Image(image)
  local size = image_sizes[image.src]
  if size == nil then
    return nil
  end
  for key, value in pairs(size) do
    image.attributes[key] = value
  end
  return image
end

function Pandoc(document)
  local body = pandoc.List()
  local found_abstract = false

  for _, block in ipairs(document.blocks) do
    if not found_abstract then
      if block.t == "Header" and stringify(block.content) == "Abstract" then
        found_abstract = true
        body:insert(title_block)
      end
    else
      if block.t == "Header" and stringify(block.content) == "References" then
        body:insert(pandoc.RawBlock("latex", "\\clearpage\n\\begingroup\n\\fontsize{7.45}{8.35}\\selectfont"))
        body:insert(block)
      elseif block.t == "Header" and stringify(block.content) == "Limitations and Dual-Use / Ethical Considerations" then
        body:insert(pandoc.RawBlock("latex", "\\endgroup\n\\clearpage"))
        body:insert(block)
      else
        body:insert(block)
      end
    end
  end

  document.blocks = body
  return document
end
