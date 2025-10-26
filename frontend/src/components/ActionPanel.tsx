type Action = {
    /** Optional card information for the action */
    card?: {
        name: string;
    };
    /** Source location for the action */
    src: string;
    /** Destination location for the action */
    dst: string;
};

type Props = {
    /** Array of available actions to display */
    actions: Action[];
};

/**
 * Component that displays a list of available game actions.
 * Shows card actions with source/destination information or pass actions.
 *
 * @param props - Component props
 * @returns JSX element containing the action panel
 */
export function ActionPanel({ actions }: Props) {
    return (
        <div className="border border-gray-200 rounded-lg p-4 bg-white">
            <h2 className="text-sm font-medium mb-3 text-gray-900">Available Actions</h2>
            {actions.length === 0 ? (
                <div className="text-gray-500 text-sm">No actions available</div>
            ) : (
                <div className="space-y-2">
                    {actions.map((action, i) => (
                        <div key={i} className="p-2 bg-gray-50 rounded text-sm text-gray-700">
                            {action.card?.name ? `${action.card.name} (${action.src} → ${action.dst})` : 'Pass'}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}