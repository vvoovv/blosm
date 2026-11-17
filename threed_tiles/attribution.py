"""Persistent import credits; display reads never modify Blender data."""
import bpy


class BlosmImportAttribution(bpy.types.PropertyGroup):
    collection: bpy.props.PointerProperty(
        type=bpy.types.Collection,
        name='Imported collection',
        description='Collection associated with these 3D Tiles credits',
    )
    # Native custom properties hold copyright strings, occurrence counts and
    # provider credits. They are saved with the registry in the blend file.


def live_collection(entry):
    """Check the stored reference only; unlinked/hidden collections still count."""
    try:
        collection = entry.collection
        if collection is not None:
            collection.name  # Reject a reference invalidated by ID deletion.
            return collection
    except ReferenceError:
        pass
    return None


def _strings(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [text.strip() for text in value if isinstance(text, str) and text.strip()]


def get_copyright(settings):
    """Build the current message without pruning entries or writing a cache."""
    counts = {}
    # Older blend files have a scene-wide message with no collection mapping.
    # Keep it as a legacy contribution; never copy new registry credits here.
    legacy = settings.get('copyright', '')
    if isinstance(legacy, str):
        for text in legacy.split(';'):
            text = text.strip()
            if text:
                counts[text] = 1

    provider_credits = []
    for entry in settings.importAttributions:
        if live_collection(entry) is None:
            continue
        occurrences = entry.get('counts', ())
        for index, text in enumerate(_strings(entry.get('copyrights', ()))):
            count = occurrences[index] if index < len(occurrences) else 1
            counts[text] = counts.get(text, 0) + (count if isinstance(count, int) and count > 0 else 1)
        provider_credits.extend(_strings(entry.get('providerCredits', ())))

    notices = sorted(counts, key=counts.get, reverse=True)
    # Preserve complete notices: different text from the same holder is distinct.
    return '; '.join(dict.fromkeys(notices + provider_credits))


def record_import(settings, collection, copyrights, provider_credits=()):
    """Publish committed geometry, pruning stale entries within the import undo."""
    registry = settings.importAttributions
    for index in range(len(registry) - 1, -1, -1):
        if live_collection(registry[index]) is None:
            registry.remove(index)

    entry = next((entry for entry in registry if entry.collection == collection), None)
    if entry is None:
        entry = registry.add()
        entry.collection = collection
    entry['copyrights'] = list(copyrights)
    entry['counts'] = list(copyrights.values())
    entry['providerCredits'] = list(dict.fromkeys(_strings(list(provider_credits))))
